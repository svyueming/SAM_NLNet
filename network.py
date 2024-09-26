from utils.Single_function import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import ShadowFormer
from SAM.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from SAM.SAMAUG import SAMAug
import torch.nn.functional as F

# CFNet
class CrossETDS(nn.Module):
    ''' ETDS arch
    args:
        num_in_ch: the number of channels of the input image
        num_out_ch: the number of channels of the output image
        upscale: scale factor
        num_block: the number of layers of the model
        num_feat: the number of channels of the model (after ET)
        num_residual_feat: the number of channels of the residual branch
    '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat, num_residual_feat):
        super().__init__()
        assert (num_feat > num_residual_feat >= num_in_ch) and (num_out_ch == num_in_ch)

        num_feat = num_feat - num_residual_feat  #29

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)
        self.conv_residual_first = nn.Conv2d(num_in_ch, num_residual_feat, kernel_size=3, padding=1)

        backbone_convs = []
        add_backbone_convs = []
        residual_convs = []
        add_residual_convs = []
        for _ in range(num_block):
            backbone_convs.append(nn.Conv2d(num_feat, num_feat, 3, padding=1))
            add_backbone_convs.append(nn.Conv2d(num_feat, num_residual_feat, kernel_size=3, padding=1))
            residual_convs.append(nn.Conv2d(num_residual_feat, num_residual_feat, kernel_size=3, padding=1))
            add_residual_convs.append(nn.Conv2d(num_residual_feat, num_feat, kernel_size=3, padding=1))
        self.backbone_convs = nn.ModuleList(backbone_convs)
        self.add_backbone_convs = nn.ModuleList(add_backbone_convs)
        self.residual_convs = nn.ModuleList(residual_convs)
        self.add_residual_convs = nn.ModuleList(add_residual_convs)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2), 3, padding=1)
        self.conv_residual_last = nn.Conv2d(num_residual_feat, num_out_ch * (upscale**2), kernel_size=3, padding=1)

        self.upsampler = nn.PixelShuffle(upscale)

        self.init_weights(num_in_ch, upscale, num_residual_feat)

    def init_weights(self, colors, upscale, num_residual_feat):
        ''' init weights (K_r --> I, K_{r2b} --> O, K_{b2r} --> O, and repeat --> repeat(I,n)) '''
        self.conv_residual_first.weight.data.fill_(0)
        for i in range(colors):
            self.conv_residual_first.weight.data[i, i, 1, 1] = 1
        self.conv_residual_first.bias.data.fill_(0)
        for residual_conv in self.residual_convs:
            residual_conv.weight.data.fill_(0)
            for i in range(num_residual_feat):
                residual_conv.weight.data[i, i, 1, 1] = 1
            residual_conv.bias.data.fill_(0)
        for add_residual_conv in self.add_residual_convs:
            add_residual_conv.weight.data.fill_(0)
            add_residual_conv.bias.data.fill_(0)
        # for add_backbone_conv in self.add_backbone_convs:
        #     add_backbone_conv.weight.data.fill_(0)
        #     add_backbone_conv.bias.data.fill_(0)
        self.conv_residual_last.weight.data.fill_(0)
        for i in range(colors):
            for j in range(upscale**2):
                self.conv_residual_last.weight.data[i * (upscale**2) + j, i, 1, 1] = 1
        self.conv_residual_last.bias.data.fill_(0)

    def forward(self, input1, input2):
        ''' forward '''
        x = F.relu(self.conv_first(input1))
        r = F.relu(self.conv_residual_first(input2))

        for backbone_conv, residual_conv, add_residual_conv, add_backbone_conv in zip(self.backbone_convs, self.residual_convs, self.add_residual_convs, self.add_backbone_convs):
            x, r = F.relu(backbone_conv(x) + add_residual_conv(r)), F.relu(residual_conv(r) + add_backbone_conv(x))

        x = self.upsampler(self.conv_last(x))
        r = self.upsampler(self.conv_residual_last(r))
        return x, r

class MaskAwareNet_lz(nn.Module):
    def __init__(self, opt):
        super(MaskAwareNet_lz, self).__init__()
        self.train_ps = opt.train_ps
        self.embed_dim = opt.embed_dim
        self.win_size = opt.win_size
        self.token_projection = opt.token_projection
        self.token_mlp = opt.token_mlp
        self.MF = ShadowFormer(img_size=self.train_ps, embed_dim=self.embed_dim, win_size=self.win_size,
                               token_projection=self.token_projection, token_mlp=self.token_mlp)

    def forward(self, x, SM, F_h, F_l):
        x = x.float()
        SM = SM.float()
        z = self.MF(x, SM, F_h, F_l)
        return z


class Network(nn.Module):
    def __init__(self,  opt):
        super(Network, self).__init__()
        self.depth = opt.depths
        self.gamma = opt.gamma
        self.beta = opt.beta
        #---------------------
        # CFNet
        self.ETDS = CrossETDS(num_in_ch=1, num_out_ch=1, upscale=1, num_block=4, num_feat=32, num_residual_feat=3).cuda()

        layers = []
        for _ in range(self.depth):
            single_layer = CSMRI_Onelayer(opt)
            layers.append(single_layer)  # 封装
        self.net = nn.Sequential(*layers)  # 容器

    def forward(self, zf, y, mask, EA, gt):
        x = Variable(zf.cuda())
        y_gpu = Variable(y.cuda())
        mask_gpu = Variable(mask.cuda())
        total_loss = 0
        x = x.float()
        EA = EA.float()
        F_h, F_l = self.ETDS(EA, zf)

        for i in range(self.depth):
            beta_input = self.beta * (self.gamma ** i)
            x, stageloss = self.net[i](x, y_gpu, mask_gpu, beta_input, EA, gt, F_h=F_h, F_l=F_l)
            total_loss += stageloss

        return x, total_loss

class CSMRI_Onelayer(nn.Module):
    def __init__(self,opt):
        super(CSMRI_Onelayer, self).__init__()
        self.prior_net = MaskAwareNet_lz(opt).cuda()
    def forward(self, x, y_gpu, mask_gpu, beta, EA, gt, F_h, F_l):
        #####先验步骤
        x_hat = self.prior_net(x, EA, F_h, F_l)
        x_hat = torch.clamp(x_hat, 0., 1.)

        #####数据保真步骤
        F_x_hat = torch.fft.fft2(x_hat)
        rec = (y_gpu * mask_gpu + beta * (F_x_hat * mask_gpu) + (1 + beta) * F_x_hat * (1 - mask_gpu)) / (1 + beta)
        rec = abs(torch.fft.ifft2(rec))
        stageloss = F.mse_loss(rec, gt, size_average=False)
        return rec, stageloss



class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint='SAM/checkpoints/sam_vit_b_01ec64.pth').cuda()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, crop_nms_thresh=0.5, box_nms_thresh=0.5, pred_iou_thresh=0.5)

    def forward(self, x):
        pic = x * 255
        pic = pic.squeeze(1).squeeze(0).unsqueeze(-1).cpu().detach().numpy().astype(np.uint8)
        pic_3c = np.tile(pic, (1,1,3))
        EA = SAMAug(pic_3c, self.mask_generator) / 255
        EA_tensor = torch.from_numpy(EA).unsqueeze(0).unsqueeze(0).cuda()
        return EA_tensor


class IntroduceSAM(nn.Module):
    def __init__(self):
        super(IntroduceSAM, self).__init__()
        #---------------------
        self.SAM = SAM().cuda()

    def forward(self, zf):
        x = Variable(zf.cuda())
        EA = self.SAM(x)

        return  EA
