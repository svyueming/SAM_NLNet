import argparse
import torch.optim as optim
import random
import logging
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import torch.nn as nn
import os
import numpy as np
import torch
from os.path import join
import time
import torch.utils.data as udata
import logging
import math
import sympy
import sys
from torch.nn.modules.loss import _Loss
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


# 随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    # torch.cuda.manual_seed_all(seed)  # all gpus

# mask
def Radial_Line_Sensing_numpy(L):
    h = 256
    w = 256
    if 1:
        aperture = (math.pi / 180) * 180
        direction = (math.pi / 180) * 0
    S = LineMaskLimitedAngle_numpy(L, h, aperture, direction)
    SS = np.ravel(S, 'F')
    P = np.where(SS == 1)

    return S, P

def LineMaskLimitedAngle_numpy(L, n, aperture, direction):
    if (math.pi - aperture) > (aperture / L):
        thc = np.linspace(-direction - aperture / 2, -direction + aperture / 2, L)
    else:
        thc = np.linspace(-direction - math.pi / 2, -direction + math.pi / 2 - math.pi / L, L)
    thc = thc % math.pi
    S = np.zeros((n, n))
    for ll in range(L):
        if ((thc[ll] <= math.pi / 4) or (thc[ll] > 3 * math.pi / 4)):
            yr = (np.round(
                (math.tan(thc[ll]) * np.array(range(-n // 2 + 1, n // 2, 1)) + n // 2 + 1).astype(float))).astype(int)
            for nn in range(n - 1):
                S[yr[nn] - 1, nn + 1] = 1
        else:
            xc = (np.round(
                (sympy.cot(thc[ll]) * np.array(range(-n // 2 + 1, n // 2, 1)) + n // 2 + 1).astype(float))).astype(int)
            for nn in range(n - 1):
                S[nn + 1, xc[nn] - 1] = 1
    S = np.fft.ifftshift(S)
    return S

def gen_mask(num_radial):  # ori_image:(256,256),生成的mask也为（256，256）
    mask, P = Radial_Line_Sensing_numpy(num_radial)
    return mask

# log
def init_logger(argdict):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(join(argdict.out_dir, 'log.txt'), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("Arguments: ")
    for k in argdict.__dict__:
        logger.info("\t{}: {}".format(k, argdict.__dict__[k]))
    return logger

class Logger(object):
    def __init__(self, stream=sys.stdout):
        output_dir = "result_log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)
        self.terminal = stream
        self.log = open(filename, 'a+')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

# loss
def Myl2_reg_ortho(W):
    cols = W[0].numel()  #####numel()函数：返回数组中元素的个数121
    cols=min(cols, W[0,0,:,:].numel()) # lqs 121
    w1 = W.view(-1,cols)   # 121 121
    wt = torch.transpose(w1,0,1)   ###这步就是求转置
    m  = torch.matmul(wt,w1)   ## W1t*W1
    ident = torch.eye(cols,cols).cuda()   # 121 * 121的I
    w_tmp = m - ident      ##### W1t-W1 - I
    l2_reg=torch.norm(w_tmp,2)**2    ##### 二范数的平方
    return l2_reg

class MSE_WWTI(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum') + lamuda * ||WtW - I||_F^2
    The backward is defined as: input-target
    """
    def __init__(self, size_average=False, reduce=True):
        super(MSE_WWTI, self).__init__(size_average, reduce)

    def forward(self, input, target, W1, lamuda):
        firstterm = nn.functional.mse_loss(input, target, size_average=False, reduce=True)
        secondterm = lamuda * Myl2_reg_ortho(W1)
        total_loss = firstterm + secondterm
        return total_loss

# 其他
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print('Total number of parameters: %d' % num_params)
    return num_params

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_PSNR_ssim(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    SSIM = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        SSIM += compare_ssim(Iclean[i,i,:,:], Img[i,i,:,:], data_range=1)
    return (PSNR/Img.shape[0]), (SSIM/Img.shape[0])

def CtoT(data):
    data = torch.cat((data.real, data.imag), dim=1)
    return data

def TtoC(data):
    data = torch.complex(data[:, 0:121, :, :], data[:, 122:243, :, :])  # 1 1 256 256
    return data

def filter_1(x):
    rshape = x.shape
    tmax, _ = x.view(rshape[0], rshape[1], -1).max(2)
    tmin, _ = x.view(rshape[0], rshape[1], -1).min(2)
    tmax = tmax. view(rshape[0], rshape[1], 1, 1)
    tmin = tmin. view(rshape[0], rshape[1], 1, 1)
    x = (x-tmin) / (1e-10 + tmax -tmin)
    out = x[:,0:1, :, :]
    return out

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))

