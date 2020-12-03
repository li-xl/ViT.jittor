# Vision Transformer - Jittor

Jittor implementation of <a href="https://openreview.net/pdf?id=YicbFdNTTy">Vision Transformer</a>.
Pytorch code is from <a href="https://github.com/lucidrains/vit-pytorch">lucidrains</a> and <a href='https://github.com/rwightman/pytorch-image-models'>Ross Wightman </a>. 

Lucidrains' code is a simple way to achieve SOTA in vision classification with only a single transformer encoder, in Pytorch. And Ross Wightman'code can load the pretrained models of <a href="https://openreview.net/pdf?id=YicbFdNTTy">Vision Transformer</a>.

In order to distinguish between them, we are called vit_v1 and vit respectively. 

### Time Performance
Vit v1 (Lucidrains' code)

|Framework| Test| Train|
|---|---|---|
|Pytorch|7s|29s|
|Jittor|6s|24s|
| Speed|1.17|1.21|

ViT ( Ross Wightman'code)
|Framework| Test| Train|
|---|---|---|
|Pytorch|1.007s|0.388s|
|Jittor| | 0.325s|
| Speed| |1.19|

### Usage
You can use *train_vit_v1.py* to train a single transformer encoder (Lucidrains' code).

And you can use *train.py* and *validate.py* totrain and test with pretrained models.


