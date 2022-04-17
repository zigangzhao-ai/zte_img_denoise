import os
import numpy as np
import rawpy
import torch
import skimage.metrics
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from .data_augment import PairCompose, PairToTensor, PairRandomCrop, PairRandomHorizontalFilp
from .data_augment import normalization, read_image, random_noise_levels, add_noise
import random


class DngDataset(Dataset):

    def __init__(self, image_dir, transform=None, is_eval=False):
        self.image_dir = image_dir
        self.input_list = os.listdir(os.path.join(image_dir, 'noisy'))
        self.ground_list = os.listdir(os.path.join(image_dir, 'ground_truth'))
        
        self._check_image(self.input_list)
        self.input_list.sort()
        self._check_image(self.ground_list)
        self.ground_list.sort()
        self.transform = transform
        self.is_test = is_eval

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        black_level = 1024
        white_level = 16383
        image, height, width = read_image(os.path.join(self.image_dir, 'noisy', self.input_list[idx]))

        image = normalization(image, black_level, white_level)
        image = torch.from_numpy(image).float()  
        ##add noise
        noise = random.random() < 0.5
        if noise:        
            # shot_noise, read_noise = random_noise_levels()
            # image  = add_noise(image, shot_noise, read_noise)
            image = add_noise(image)
        image = image.view(-1, height//2, width//2, 4).permute(0, 3, 1, 2)
     
        # image = torch.from_numpy(np.transpose(image.reshape(-1, height//4, width//4, 16), (0, 3, 1, 2))).float()
        label, height, width = read_image(os.path.join(self.image_dir, 'ground_truth', self.ground_list[idx]))
        label = normalization(label, black_level, white_level)
        label = torch.from_numpy(np.transpose(label.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] == 'DS_Store':
                continue
            if splits[-1] not in ['dng']:
                raise ValueError

class DngTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = os.path.join(image_dir,'test')
        # self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        self.input_list = os.listdir(self.image_dir)
        self._check_image(self.input_list)
        self.input_list.sort()
        self.transform = transform


    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        black_level = 1024
        white_level = 16383
        image, height, width = read_image(os.path.join(self.image_dir,self.input_list[idx]))
        image = normalization(image, black_level, white_level)
        image = torch.from_numpy(np.transpose(image.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()
        name = self.input_list[idx]
        return image,name

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] == 'DS_Store':
                continue     
            if splits[-1] not in ['dng']:
                raise ValueError


def train_dataloader(path, batch_size=4, num_workers=0, use_transform=True):
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop((1736,2312)),
                PairRandomHorizontalFilp(p=0.5),
            ]
        )
    dataloader = DataLoader(
        DngDataset(os.path.join(path, 'train'), transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DngTestDataset(path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DngDataset(os.path.join(path, 'valid'), is_eval=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader