from __future__ import print_function
import argparse
import os
import time
import torch
import sys
import torch.nn.functional as F
import PIL
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from math import ceil
from utils.utils import batch_PSNR, init_logger
from utils.SSIM import SSIM
from utils import utils_image
from utils.Single_function import *
from network import IntroduceSAM
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设备号 默认为0

class Dataset(udata.Dataset):
    def __init__(self, t='train'):
        self.type = t
        if self.type == 'train':
            self.data = np.load("data/brain_4000_norm.npy")
        elif self.type == 'val':
            self.data = np.load("data/brain_500_norm.npy")
        elif self.type == 'test':
            self.data = np.load("data/single_brain_30.npy")
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        if self.type == 'train':
            self.data = np.load("data/brain_4000_norm.npy")
        elif self.type == 'val':
            self.data = np.load("data/brain_500_norm.npy")
        elif self.type == 'test':
            self.data = np.load("data/single_brain_30.npy")
        x = self.data[index]
        return x


total = []
sigma_hot = 0.1
dataset_train = Dataset(t='train')
dataset_val = Dataset(t='val')
loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=1, shuffle=False)
loader_val = DataLoader(dataset=dataset_val, num_workers=0, batch_size=1, shuffle=False)
model = IntroduceSAM().cuda()
model.eval()

for i, image_train in enumerate(loader_val):
    image_train = image_train.float()
    image_train_np = image_train.squeeze(0).squeeze(0).cpu().detach().numpy()

    # 创建采样矩阵
    mask = gen_mask(55)
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(1).float().cuda()
    mask_input = mask.cuda()
    # 创建噪声矩阵
    noise = sigma_hot / 255 * torch.randn(256, 256) + sigma_hot / 255 * torch.randn(256, 256) * (1.j)
    noise_input = noise.unsqueeze(0).unsqueeze(1).cuda()
    # 数据处理
    train_gt_fft = torch.fft.fft2(image_train).cuda()
    train_gt_fft_mask = train_gt_fft * mask_input
    train_gt_fft_mask_noise = train_gt_fft_mask + noise_input
    y = train_gt_fft_mask_noise.cuda()
    gt = image_train.float().cuda()
    zf = torch.fft.ifft2(y).cuda()
    abs_zf = abs(zf)

    with torch.no_grad():
        x_rec = model(abs_zf)  # 重建图像
    EA_np = x_rec.squeeze(0).squeeze(0).cpu().detach().numpy()
    a = np.stack((image_train_np, EA_np), axis=0)
    total.append(a)
    print(len(total))

np.save('data/brain500NormandSAM.npy', total)





