import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train2500')
    
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop((480,480)),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test2000')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def custom_test_dataloader(path, batch_size=1, num_workers=0):
    # image_dir = os.path.join(path, 'test2000')
    image_dir = path
    dataloader = DataLoader(
        CustomDeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    transform = PairCompose(
            [
                PairRandomCrop((480,480)),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset((os.path.join(path, 'test2000')),transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'Snow/'))
#        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'Snow', self.image_list[idx]))
        # label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx].split('.')[0]+'.jpg'))#srrs+jpg
        label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label


class CustomDeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        # self.image_list = os.listdir(os.path.join(image_dir, 'Snow/'))
        self.image_list = os.listdir(image_dir)
#        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.image_list[idx]))
        # label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx].split('.')[0]+'.jpg'))#srrs+jpg
        # label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx]))

        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)
            # label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, name
        return image
