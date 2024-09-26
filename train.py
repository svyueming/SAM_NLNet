from __future__ import print_function
import torch
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设备号 默认为0
import time
import sys
import torch.nn.functional as F
import PIL
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from Dataset import lzDataset
from math import ceil
from utils.utils import batch_PSNR, init_logger
from utils.SSIM import SSIM
from utils import utils_image
from utils.Single_function import *
from network import Network



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def main(opt):
    #——————————————————— use_gpu=1——————————————————————————
    sigma_hot = 0.1
    Pcount_psnr = []
    Pcount_ssim = []
    #—————————————————— create path————————————————————————
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)

    logger = init_logger(opt)
    sys.stdout = Logger(sys.stdout)
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    seed = opt.manualSeed
    model = Network(opt).cuda()
    print_network(model)
    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone, gamma=opt.lr_decay)
    #——————————————————— load dataset ———————————————
    dataset_train = lzDataset(t='train')
    dataset_val = lzDataset(t='val')
    loader_train = DataLoader(dataset=dataset_train, num_workers=2, batch_size=opt.batchSize, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=2, batch_size=opt.batchSize, shuffle=True)
    #———————————— Resume training or start a new————————————
    if opt.resume_training:
        resumef = os.path.join(opt.out_dir, 'ckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            new_epoch = opt.num_epochs
            new_milestone = opt.milestone
            opt = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            best_psnr = training_params['best_psnr']
            opt.num_epochs = new_epoch
            opt.milestone = new_milestone
            print("=> loaded checkpoint '{}' (epoch {})".format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint['optimizer']['param_groups']))
            print("==> checkpoint['training_params']")
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = vars(checkpoint['args'])
            print("==> checkpoint['args']")
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))
            opt.resume_training = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}".format(resumef))
    else:
        training_params = {}
        start_epoch = 0
        best_psnr = 0
    # —————————————————      training   ——————————————————————————————
    for epoch in range(start_epoch, opt.num_epochs): # epoch=[0--300)=0-299
        time_start = time.time()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(current_lr)
        # ———————————————————— train stage————————————————————————————————————
        epoch_loss_train = 0
        epoch_samples_train = 0
        psnr_train_average = 0
        ssim_train_average = 0
        countt = 0

        tic = time.time()
        optimizer.zero_grad()

        for i, image_batch in enumerate(loader_train):
            countt = countt + 1
            image_train = image_batch[:, :1, :, :]
            image_train = image_train.float()
            SAMmap = image_batch[:, 1:2, :, :].cuda()

            model.train()
            optimizer.zero_grad()
            # 创建采样矩阵
            mask = gen_mask(opt.num_radial)
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
            # plt.imshow(abs_zf.squeeze(0).squeeze(0).cpu().detach().numpy() * 255)
            # plt.show()
            # 输入model
            x_hat, totalloss = model(abs_zf, y, mask_input, SAMmap, gt)
            loss = F.mse_loss(x_hat, gt, size_average=False) + opt.loss_weight * totalloss
            # back propagation
            loss.backward()
            optimizer.step()

            model.eval()
            epoch_loss_train += loss.data.cpu().numpy()
            epoch_samples_train += len(y)
            x_rec = x_hat  # 1 1 256 256
            psnr_train, ssim_train = batch_PSNR_ssim(x_rec, gt, 1.)
            psnr_train_average += psnr_train
            ssim_train_average += ssim_train
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f, SSIM_train: %.4f, Current_lr: %.8f" % (
            epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train, ssim_train, current_lr))
            sys.stdout.flush()

        epoch_loss_train /= epoch_samples_train
        train_epoch_loss = epoch_loss_train
        psnr_train_average /= countt
        ssim_train_average /= countt
        print("train_epoch_loss = {}, psnr_train_average = {}, ssim_train_average = {}".format(train_epoch_loss, psnr_train_average, ssim_train_average))
        scheduler.step()

        #—————————————————————————— evaluation stage——————————————————————————
        model.eval()
        epoch_loss_val = 0
        epoch_samples_val = 0
        count = 0
        for ii, val_batch in enumerate(loader_val):
            count = count + 1
            image_val = val_batch[:, :1, :, :]
            image_val = image_val.float()
            sammap = val_batch[:, 1:2, :, :].cuda()
            # mask
            mask = gen_mask(opt.num_radial)  # 256 256
            mask = torch.tensor(mask).unsqueeze(0).unsqueeze(1).float().cuda()
            mask_val = mask.cuda()
            # noise
            noise = sigma_hot / 255 * torch.randn(256, 256) + sigma_hot / 255 * torch.randn(256, 256) * (1.j)
            noise_val = noise.unsqueeze(0).unsqueeze(1).cuda()
            #
            val_gt_fft = torch.fft.fft2(image_val).cuda()
            val_gt_fft_mask = val_gt_fft * mask_val
            val_gt_fft_mask_noise = val_gt_fft_mask + noise_val
            y = val_gt_fft_mask_noise.cuda()
            gt = image_val.float().cuda()
            zf = torch.fft.ifft2(y).cuda()
            abs_zf = abs(zf)
            with torch.no_grad():
                x_hat, tloss = model(abs_zf, y, mask_val, sammap, gt)
                loss = F.mse_loss(x_hat, gt, size_average=False) +  opt.loss_weight * tloss
                epoch_loss_val += loss.data.cpu().numpy()
            epoch_samples_val += len(y)
            x_rec = x_hat
            psnr_val0, ssim_val0 = batch_PSNR_ssim(x_rec, gt, 1.)
            print("\n[epoch %d][%d/%d]  PSNR_val: %.4f,  SSIM_val: %.4f" % (
            epoch + 1, count, len(loader_val), psnr_val0, ssim_val0))
            Pcount_psnr.append(psnr_val0)
            Pcount_ssim.append(ssim_val0)

        psnr_val = np.mean(Pcount_psnr)
        ssim_val = np.mean(Pcount_ssim)
        time_end = time.time()
        cur_time = time_end - time_start
        print("\n[epoch %d] PSNR_val: %.4f, SSIM_val: %.4f, Time： %.2f" % (epoch + 1, psnr_val, ssim_val, cur_time))
        sys.stdout.flush()
        epoch_loss_val /= epoch_samples_val
        val_epoch_loss = epoch_loss_val
        print("val_epoch_loss=", val_epoch_loss)
        logger.info(
            "\tcurrent_epoch:{}  PSNR_val:{:.4f}  SSIM_val:{:.4f}  best_psnr:{:.4f}  TRAINLOSS:{:.4f}  VALLOSS:{:.4f}  currentlr:{:.4f}  seed：{}".format(
                epoch + 1, psnr_val, ssim_val, best_psnr, train_epoch_loss, val_epoch_loss, current_lr, seed))
        sys.stdout.flush()

        #————————————————————————————— save models————————————————————————
        training_params['start_epoch'] = epoch + 1
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            training_params['best_psnr'] = best_psnr
            torch.save(model.state_dict(), f'{opt.out_dir}/{opt.model_name}.pth')


        if (epoch + 1) % 1 == 0:  # 输出结果
            torch.save(model.state_dict(), f'{opt.out_dir}/{epoch + 1}.pth')
        toc = time.time()  # 获取当前时间戳
        print('This epoch take time {:.2f}'.format(toc - tic))

        #—————————————————————————— save current modelformer and checkpoint————————————————————
        save_dict = {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     'training_params': training_params,
                     'args': opt}
        torch.save(save_dict, os.path.join(opt.out_dir, 'ckpt.pth'))  # 保存current model

    print('Reach the maximal epochs! Finish training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Second Point")
    parser.add_argument("--resume_training", "--r", default=False, help="resume training from a previous checkpoint")
    # parser.add_argument("--resume_training", "--r", default=True, help="resume training from a previous checkpoint")
    #-----------------------常用参数-----------------------#
    parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results dir path", default='Knee-R-55')
    parser.add_argument("--num_radial", type=int, help="15,25,55,85", default=55) # 15-0.05 25-0.1 55-0.2 85-0.3
    parser.add_argument("--beta", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=1.02)
    parser.add_argument("--loss_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0002, help="Initial learning rate")
    parser.add_argument("--depths", type=int, dest="depths", help="The depth of the network", default=8)
    #----------------------------------------------------#
    parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
    parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
    parser.add_argument('--win_size', type=int, default=4, help='window size of self-attention')
    parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
    #----------------------------------------------------#
    parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.", default='best_model')
    parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--milestone", type=int, default=[30], help="When to decay learning rate; should be less than epochs")
    parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.1)
    opt = parser.parse_args()
    main(opt)





