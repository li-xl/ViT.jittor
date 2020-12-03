#coding=utf-8
import torch
import glob
import os
import numpy as np
import jittor as jt 
import jittor.transform as transforms
from jittor.dataset import Dataset
from jittor import nn,optim
from tqdm import tqdm 
from PIL import Image
from sklearn.model_selection import train_test_split
from models.vit_v1 import ViT
import random


# settings
batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7 
seed = 42
num_workers = 4

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    jt.set_seed(seed)


seed_everything(seed)

# use cuda
jt.flags.use_cuda = 1

## Load Data
train_dir = '/mnt/disk/lxl/dataset/kaggle/dogs_vs_cats/train'
test_dir = '/mnt/disk/lxl/dataset/kaggle/dogs_vs_cats/test'

train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

labels = [path.split('/')[-1].split('.')[0] for path in train_list]

## Split
train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=42)

print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")

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


train_data = CatsDogsDataset(train_list, transform=transform,batch_size=batch_size, shuffle=False, num_workers=num_workers)
valid_data = CatsDogsDataset(valid_list, transform=transform,batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_data = CatsDogsDataset(test_list, transform=transform,batch_size=batch_size, shuffle=False, num_workers=num_workers)

print(len(train_data))
print(len(valid_data))

model = ViT(
      dim=128,
      image_size=224,
      patch_size=32,
      num_classes=2,
      depth=12,
      heads=8,
      mlp_dim=128
)

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
