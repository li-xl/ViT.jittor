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
#### ViT V1(lucidrains')
**Dataset:** [Cat & Dog](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

**Model:**
```python
from models.vit_v1 import ViT
model = ViT(
      dim=128,
      image_size=224,
      patch_size=32,
      num_classes=2,
      depth=12,
      heads=8,
      mlp_dim=128
)
```
For cuda usage,we can use it with this.
```python
jt.flags.use_cuda = 1
```
**Dataloader:**
```python
import jittor.transform as transforms
from jittor.dataset import Dataset
## Image Augumentation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomCropAndResize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
## Load Datasets
class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None,batch_size=1,shuffle=False,num_workers=0):
        super(CatsDogsDataset,self).__init__(batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        self.file_list = file_list
        self.transform = transform
        self.total_len=len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label
```
**Train & Test**
```python
### Training
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_data):
        output = model(data)
        loss = criterion(output, label)
        optimizer.step(loss)

        acc = (output.argmax(dim=1)[0] == label).float().mean()
        epoch_accuracy += acc / len(train_data)
        epoch_loss += loss / len(train_data)

    with jt.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in tqdm(valid_data):
            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1)[0] == label).float().mean()
            epoch_val_accuracy += acc / len(valid_data)
            epoch_val_loss += val_loss / len(valid_data)
    jt.sync_all(True)
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss.item():.4f} - acc: {epoch_accuracy.item():.4f} - val_loss : {epoch_val_loss.item():.4f} - val_acc: {epoch_val_accuracy.item():.4f}\n"
    )
```

#### Vit(Ross Wightman')

**Dataset:**  [Imagenet](http://www.image-net.org/)

**Model Names:**
vit_huge_patch32_384,
vit_huge_patch16_224,
vit_large_patch32_384,
vit_large_patch16_384,
vit_large_patch16_224,
vit_base_patch32_384,
vit_base_patch16_384,
vit_base_patch16_224,
vit_small_patch16_224

**Train & Validate:**
Use *train.py* to train your ViT and *validate.py* to validate your ViT.

Use this to set configs.
```python
jt.flags.use_cuda = 1

model_name = 'vit_base_patch16_224'
lr = 0.001
train_dir = '/data/imagenet/train'
eval_dir = '/data/imagenet/val'
batch_size = 32
input_size = 224
num_workers = 4
hflip = 0.5
ratio = (0.75,1.3333333333333333)
scale = (0.08,1.0)
train_interpolation = 'random'
num_epochs = 8
```

You can use this to create your model and dataset.
```python
model  = create_model('vit_base_patch16_224',pretrained=True,num_classes=1000)
criterion = nn.CrossEntropyLoss()
     
dataset = create_val_dataset(root='/data/imagenet',batch_size=bs,num_workers=4,img_size=224)

```

