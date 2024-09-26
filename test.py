from scipy.io import savemat
from network import Network
from utils.Single_function import *
from Dataset import lzDataset
import torch
import scipy.io as scio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设备号 默认为0

# 将模型加载到当前设备（通常是CUDA:0）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
set_seed(0)

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))

def main(opt):
    num_radial = opt.num_radial
    sigma = 0.1
    Pcount_psnr = []
    Pcount_ssim = []
    Pcount_rmse = []
    # 加载训练好的模型
    model = Network(opt).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.out_dir, 'best_model.pth'), map_location=device))
    model.eval()

    dataset_val = lzDataset(t='test')
    loader_val = DataLoader(dataset=dataset_val, num_workers=0, batch_size=1, shuffle=False)
    for ii, image_batch in enumerate(loader_val, 0):
        print('Image  {}'.format(ii))


        image_test = image_batch[:, :1, :, :]
        image_test = image_test.float()
        sammap = image_batch[:, 1:2, :, :].cuda()
        # mask
        # if opt.num_radial == 15:
        #     data = scio.loadmat('data/final_mask/Q_cartesian_0.05.mat')
        #     mask = np.transpose(data['mask'][:])
        #     mask = np.fft.ifftshift(mask)
        #     mask = torch.FloatTensor(mask).cuda()
        #
        # elif opt.num_radial == 25:
        #     data = scio.loadmat('data/final_mask/Q_cartesian_0.1.mat')
        #     mask = np.transpose(data['mask'][:])
        #     mask = np.fft.ifftshift(mask)
        #     mask = torch.FloatTensor(mask).cuda()
        #
        # elif opt.num_radial == 55:
        #     data = scio.loadmat('data/final_mask/Q_cartesian_0.2.mat')
        #     mask = np.transpose(data['mask'][:])
        #     mask = np.fft.ifftshift(mask)
        #     mask = torch.FloatTensor(mask).cuda()
        #
        # elif opt.num_radial == 85:
        #     data = scio.loadmat('data/final_mask/Q_cartesian_0.3.mat')
        #     mask = np.transpose(data['mask'][:])
        #     mask = np.fft.ifftshift(mask)
        #     mask = torch.FloatTensor(mask).cuda()


        mask = gen_mask(num_radial)
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(1).float()
        mask_test = mask.cuda()
        # noise
        noise = sigma/255 * torch.randn(256, 256) + sigma /255 * torch.randn(256, 256) * (1.j)
        noise_val = noise.unsqueeze(0).unsqueeze(1).cuda()
        #
        image_test = torch.tensor(image_test).float()
        image_test_fft = torch.fft.fft2(image_test).cuda()
        image_test_fft_mask = image_test_fft * mask_test
        image_test_fft_mask_noise = image_test_fft_mask + noise_val
        y = image_test_fft_mask_noise.cuda()
        gt = image_test.cuda()
        y_input = y
        zf = torch.fft.ifft2(y_input).cuda()
        zf_input = abs(zf)
        with torch.no_grad():
            x_hat, tloss = model(zf_input, y_input, mask_test, sammap, gt)
        x_rec = x_hat
        ###########save#############
        # rec_play = x_rec.squeeze(1).squeeze(0)
        # rec_play = rec_play.detach().cpu().numpy() * 255
        # cv2.imwrite(r'/root/autodl-tmp/results/kneeR{}/{}_rec.png'.format(num_radial,ii), rec_play)
        # savemat(os.path.join(r'/root/autodl-tmp/results/kneeR{}'.format(num_radial), f'knee_radial_{ii}_{num_radial}.mat'),{'rec_im': rec_play})

        psnr_val0, ssim_val0 = batch_PSNR_ssim(x_rec, gt, 1.)
        rmse = compute_RMSE(x_rec, gt)
        print("PSNR_val: {}, SSIM_val: {}, rmse_val: {}".format(psnr_val0, ssim_val0, rmse))
        Pcount_psnr.append(psnr_val0)
        Pcount_ssim.append(ssim_val0)
        Pcount_rmse.append(rmse)

    psnr_val = np.mean(Pcount_psnr)
    ssim_val = np.mean(Pcount_ssim)
    rmse_val = np.mean(Pcount_rmse)
    print("PSNR_val_mean: {}, SSIM_val_mean: {}, rmse_val_mean: {}".format(psnr_val, ssim_val, rmse_val))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Second Point")
    #-----------------------常用参数-----------------------#
    parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results dir path", default='Knee-R-55')
    parser.add_argument("--num_radial", type=int, help="15,25,55,85", default=55)
    parser.add_argument("--beta", type=float, default=0.0001)  # deep 原0.0001
    parser.add_argument("--gamma", type=float, default=1.02)
    parser.add_argument("--depths", type=int, dest="depths", help="The depth of the network", default=8)
    #----------------------------------------------------#
    parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
    parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
    parser.add_argument('--win_size', type=int, default=4, help='window size of self-attention')
    parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
    #----------------------------------------------------#
    opt = parser.parse_args()  # 使用parse_args()解析添加的参数
    main(opt)
