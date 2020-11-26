#coding=utf-8
import jittor as jt 
from jittor import nn
from jittor.misc import _pair
from functools import partial
import math
import numpy as np

from .utils import trunc_normal,load_pretrained,_conv_filter
from .config import default_cfgs

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x):
        b,n,c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c// self.num_heads).permute(2, 0, 3, 1, 4)
        
        q,k,v = qkv[0],qkv[1],qkv[2]

        # attn = nn.bmm(q,k.transpose(0,1,3,2))*self.scale
        attn = nn.bmm_transpose(q, k)*self.scale
        
        attn = nn.softmax(attn,dim=-1)

        attn = self.attn_drop(attn)

        out = nn.bmm(attn,v)
        out = out.transpose(0,2,1,3).reshape(b,n,c)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if self.drop_prob == 0. or not self.is_training():
            return x
        keep_prob = 1-self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + jt.random(shape, dtype=x.dtype)
        random_tensor = jt.floor(random_tensor)  # binarize
        output = (x / keep_prob) * random_tensor
        return output


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block,self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, 
                              num_heads=num_heads, 
                              qkv_bias=qkv_bias, 
                              qk_scale=qk_scale, 
                              attn_drop=attn_drop, 
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, 
                       hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, 
                       drop=drop)

    def execute(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbed,self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def execute(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(0, 2, 1)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super(HybridEmbed,self).__init__()
        assert isinstance(backbone, nn.Module)
        img_size = _pair(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with jt.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.is_training()
                if training:
                    backbone.eval()
                o = self.backbone(jt.zeros((1, in_chans, img_size[0], img_size[1])))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train()
        else:
            feature_size = _pair(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def execute(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(0,2,1)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 hybrid_backbone=None, 
                 norm_layer=nn.LayerNorm):
        super(VisionTransformer,self).__init__()
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = jt.zeros((1, 1, embed_dim))
        self.pos_embed = jt.zeros((1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

        self.pos_embed = trunc_normal(self.pos_embed, std=.02)
        self.cls_token = trunc_normal(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def apply(self,fn):
        for m in self.modules():
            fn(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def execute(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        _,i,j = self.cls_token.shape
        cls_tokens = self.cls_token.expand((B, i, j))  # stole cls_tokens impl from Phil Wang, thanks

        x = jt.contrib.concat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x[:, 0])
        return x



def vit_small_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model



def vit_base_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model



def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_384']
    if pretrained:
        load_pretrained(
            model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model



def vit_base_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch32_384']
    if pretrained:
        load_pretrained(
            model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model



def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    return model



def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_384']
    if pretrained:
        load_pretrained(
            model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model



def vit_large_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch32_384']
    if pretrained:
        load_pretrained(
            model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model



def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch16_224']
    return model



def vit_huge_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch32_384']
    return model

MODELS = {
    'vit_huge_patch32_384':vit_huge_patch32_384,
    'vit_huge_patch16_224':vit_huge_patch16_224,
    'vit_large_patch32_384':vit_large_patch32_384,
    'vit_large_patch16_384':vit_large_patch16_384,
    'vit_large_patch16_224':vit_large_patch16_224,
    'vit_base_patch32_384':vit_base_patch32_384,
    'vit_base_patch16_384':vit_base_patch16_384,
    'vit_base_patch16_224':vit_base_patch16_224,
    'vit_small_patch16_224':vit_small_patch16_224
}

def create_model(name,**kwargs):
    assert name in MODELS.keys()
    return MODELS[name](**kwargs)
    
