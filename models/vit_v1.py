#coding=utf-8
import jittor as jt 
from jittor import nn 

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self,fn):
        super(Residual,self).__init__()
        self.fn = fn
    
    def execute(self,x,**kwargs):
        return self.fn(x,**kwargs)+x 


class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super(PreNorm,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn 
    def execute(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)


class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
                   nn.Linear(dim,hidden_dim),
                   nn.GELU(),
                   nn.Dropout(dropout),
                   nn.Linear(hidden_dim,dim),
                   nn.Dropout(dropout)
        )
    def execute(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim,heads=8,dropout=0.):
        super(Attention,self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim,dim*3,bias=False)
        self.to_out = nn.Sequential(
                      nn.Linear(dim,dim),
                      nn.Dropout(dropout)
        )

    def execute(self,x,mask=None):
        b,n,_ = x.shape
        h = self.heads
        q,k,v = self.to_qkv(x).chunk(3,dim=-1)
        
        q = q.reshape(b,n,h,-1)
        q = q.transpose(0,2,1,3)

        k = k.reshape(b,n,h,-1)
        k = k.transpose(0,2,1,3)

        v = v.reshape(b,n,h,-1)
        v = v.transpose(0,2,1,3)

        #b,h,n,d
        d = q.shape[-1]
        q = q.reshape(b*h,n,d)
        k = k.reshape(b*h,n,d).transpose(0,2,1)

        dots = nn.bmm(q,k).reshape(b,h,n,n)
        dots = dots*self.scale

        if mask is not None:
            mask = nn.pad(mask.flatten(1),(1,0),value=1)
            assert mask.shape[-1] == dots.shape[-1],'mask has incorrect shapes'
            mask = mask.unsqueeze(1)*mask.unsqueeze(2)
            dots.masked_fill_(~mask,float('-inf'))
            del mask
        
        attn = nn.softmax(dots,dim=-1)

        out = nn.bmm(attn.reshape(b*h,n,n),v.reshape(b*h,n,d)).reshape(b,h,n,d)
        out = out.transpose(0,2,1,3).reshape(b,n,h*d)
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,mlp_dim,dropout):
        super(Transformer,self).__init__()
        layers = []
        for _ in range(depth):
            layers.append(
                nn.Sequential(
                    Residual(PreNorm(dim,Attention(dim,heads = heads,dropout = dropout))),
                    Residual(PreNorm(dim,FeedForward(dim,mlp_dim,dropout=dropout)))
                )
            )
        self.layers = nn.Sequential(layers)

    def execute(self,x,mask = None):
        for attn,ff in self.layers:
            x = attn(x,mask=mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self,
                 *,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 transformer = None,
                 channels=3,
                 dropout = 0.,
                 emb_dropout = 0.):
        super(ViT,self).__init__()
        assert image_size % patch_size == 0, 'image sizes must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size **2
        assert num_patches > MIN_NUM_PATCHES, f'your num_patches is too small'
        
        self.patch_size = patch_size

        self.pos_embedding = jt.random((1,num_patches+1,dim),dtype='float32')
        self.patch_to_embedding = nn.Linear(patch_dim,dim)
        self.cls_token = jt.random((1,1,dim),dtype='float32')
        self.dropout = nn.Dropout(emb_dropout)
        
        if transformer is None:
            self.transformer = Transformer(dim,depth,heads,mlp_dim,dropout)
        else:
            self.transformer = transformer

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,num_classes)
        )

    def execute(self,img,mask = None):
        p = self.patch_size

        b,c,h,w = img.shape 

        # b c (h p1) (w p2) -> b (h w) (p1 p2 c)
        x = img.reshape(b,c,h//p,p,w//p,p).transpose(0,2,4,3,5,1).reshape(b,-1,p*p*c)

        x = self.patch_to_embedding(x)
        b,n,_ = x.shape

        _,i,j = self.cls_token.shape
        cls_tokens = self.cls_token.expand((b,i,j))
        x = jt.contrib.concat((cls_tokens,x),dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x,mask)

        x = self.to_cls_token(x[:,0])
        x = self.mlp_head(x)
        return x









