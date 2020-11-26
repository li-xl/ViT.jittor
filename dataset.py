import os
import re
import jittor as jt
import jittor.transform as transforms
from jittor import dataset
from PIL import Image
import math
import random
import numpy as np


IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, tuple):
            size = size[0]
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

class Resize:
    def __init__(self,img_size,interpolation):
        self.img_size = img_size
        self.interpolation = interpolation

    def __call__(self,img):
        return resize(img,self.img_size,self.interpolation)


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)

class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return transforms.crop_and_resize(img, i, j, h, w, self.size, interpolation)


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=0.9,
        interpolation=Image.BICUBIC,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)):
    crop_pct = crop_pct or 0.875

    if isinstance(img_size, tuple):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))
    return transforms.Compose([
        Resize(scale_size, interpolation),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.ImageNormalize(mean=mean,std=std)
    ])

def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        interpolation='random',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
    primary_tfl = [
        RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    final_tfl = [
            transforms.ToTensor(),
            transforms.ImageNormalize(
                mean=mean,
                std=std)
        ]
    return transforms.Compose(primary_tfl + final_tfl)



def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


class Dataset(dataset.Dataset):

    def __init__(
            self,
            root,
            transform=None,
            shuffle=False,
            num_workers=0,
            batch_size=1
            ):
        super(Dataset,self).__init__(shuffle=shuffle,num_workers=num_workers,batch_size=batch_size)
        images, class_to_idx = find_images_and_targets(root, class_to_idx=None)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.total_len = len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = jt.zeros((1,)).int32()
        return img, target

_pil_interpolation = {
    'bicubic':Image.BICUBIC,
    'nearst': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'antialias': Image.ANTIALIAS
}

def create_val_dataset(root,img_size=224,crop_pct=0.9,interpolation='bicubic',mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),shuffle=False,num_workers=0,batch_size=1):
    transform = transforms_imagenet_eval(img_size=img_size,crop_pct=crop_pct,interpolation=_pil_interpolation['bicubic'],mean=mean,std=std)
    dataset = Dataset(root,shuffle=shuffle,num_workers=num_workers,batch_size=batch_size,transform=transform)
    return dataset

def create_train_dataset(root, img_size=224,scale=None,ratio=None,hflip=0.5,vflip=0.,interpolation='random',mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),shuffle=False,num_workers=0,batch_size=1):
    transform = transforms_imagenet_train(img_size=img_size,scale=scale,hflip=hflip,vflip=vflip,interpolation=interpolation,mean=mean,std=std)
    dataset = Dataset(root,shuffle=shuffle,num_workers=num_workers,batch_size=batch_size,transform=transform)
    return dataset