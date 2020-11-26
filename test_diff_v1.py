#coding=utf-8
import jittor as jt
from jittor_utils import auto_diff

import torch
import glob

from PIL import Image 
import numpy as np 
from tqdm import tqdm 


from models.vit import ViT as jt_ViT
from models.vit_pytorch import ViT as torch_ViT

torch_model = torch_ViT(
      dim=128,
      image_size=224,
      patch_size=32,
      num_classes=2,
      depth=12,
      heads=8,
      mlp_dim=128
).cuda()

jt.flags.use_cuda=1
jt.flags.merge_loop_mismatch_threshold = 1


jt_model = jt_ViT(
      dim=128,
      image_size=224,
      patch_size=32,
      num_classes=2,
      depth=12,
      heads=8,
      mlp_dim=128
)

# # state_dict = torch.load('model.pkl')

# torch_model.load_state_dict(state_dict)
# jt_model.load_parameters(state_dict)


# image = Image.open('test.jpg')
# image = np.array(image)[None,:,:,:]
# image = image.transpose(0,3,1,2).astype(np.float32)
image = np.random.randn(64,3,224,224).astype(np.float32)

# hook = auto_diff.Hook('ViT')
# hook.hook_module(jt_model)

for i in tqdm(range(300)):
    jt_data = jt.array(image)
    jt_output = jt_model(jt_data).numpy()

# print(jt_output)

for i in tqdm(range(300)):
    torch_data = torch.tensor(image).cuda()
    torch_output = torch_model(torch_data).cpu().detach().numpy()


# assert np.allclose(jt_output,torch_output)


