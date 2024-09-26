import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
from numpy.random import RandomState
import PIL
from PIL import Image

def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data.astype(np.float32)
    data = data*255.0
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    def _augment(img):
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        return img
    return [_augment(a) for a in args]

class lzDataset(udata.Dataset):
    def __init__(self, t='train'):
        self.type = t
        if self.type == 'train':
            self.data = np.load("data/knee4000NormandSAM.npy")
        elif self.type == 'val':
            self.data = np.load("data/knee500NormandSAM.npy")
        elif self.type == 'test':
            self.data = np.load("data/knee30NormandSAM.npy")
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        if self.type == 'train':
            self.data = np.load("data/knee4000NormandSAM.npy")
        elif self.type == 'val':
            self.data = np.load("data/knee500NormandSAM.npy")
        elif self.type == 'test':
            self.data = np.load("data/knee30NormandSAM.npy")
        x = self.data[index]
        return x


class lzDataset2(udata.Dataset):
    def __init__(self, t='train'):
        self.type = t
        if self.type == 'train':
            self.data = np.load("data/brain4000NormandSAM.npy")
        elif self.type == 'val':
            self.data = np.load("data/brain500NormandSAM.npy")
        elif self.type == 'test':
            self.data = np.load("data/brain30NormandSAM.npy")
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        if self.type == 'train':
            self.data = np.load("data/brain4000NormandSAM.npy")
        elif self.type == 'val':
            self.data = np.load("data/brain500NormandSAM.npy")
        elif self.type == 'test':
            self.data = np.load("data/brain30NormandSAM.npy")
        x = self.data[index]
        return x