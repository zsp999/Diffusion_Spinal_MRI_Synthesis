import numpy as np
import matplotlib.pyplot as plt
from IPython.display import  clear_output
from IPython.display import display as dp
import random
import os
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

import sys 
sys.path.append("..") 
from model.unet_attention_False import UNetModel, SimpleDiscriminator
from model.dataset import *
from model.loss import *
from model.loss import torch_ssim

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse

spinal_train_dir = "/home/zhangsenpeng/MRI_sequence_synthesis/70MRI_1/train_spinal_MRI_1126"
spinal_test_dir = "/home/zhangsenpeng/MRI_sequence_synthesis/70MRI_1/test_spinal_MRI_1126"
spinal_train_dataset = MRI_patient_Dataset(dir_path=spinal_train_dir)#spinal_train_dir
spinal_test_dataset = MRI_patient_Dataset_fortest(dir_path=spinal_test_dir)

class Trainer(object):
    def __init__(self, net_G:nn.Module, net_D:nn.Module, train_dataloader, test_dataloader, 
                 epoches, learning_rate, use_C_Loss, use_SSIM_Loss, device,\
                logname, savename_GD, remark = "Nothing", picture_save = None ):
        self.picture_savename = 'spinal_test'
        self.epoches = epoches
        self.lr = learning_rate
        self.use_C_Loss = use_C_Loss #基线测试先不用
        self.use_SSIM_Loss = use_SSIM_Loss
        self.device = device
        self.logname = logname
        self.savename_GD = savename_GD
        self.remark = remark
        self.picture_save = picture_save
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.net_G = net_G.to(device)
        self.net_D = net_D.to(device)
        self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.optimizer_D = optim.Adam(self.net_D.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.scheduler_G = CosineAnnealingWarmRestarts(self.optimizer_G, T_0=1, T_mult=1) #开1倍扩增
        self.scheduler_D = CosineAnnealingWarmRestarts(self.optimizer_D, T_0=1, T_mult=1)#开1倍扩增
        self.gan_patchloss = nn.MSELoss() #GANloss用L2
        # 损失函数先L1再L2试试，先L2不太好 L2损失效果不好
        self.rec_loss = nn.L1Loss() #用L1损失
        self.G_rec_loss_list = []
        self.scaler = GradScaler()#混合精度浮点运算

    def get_data_from_batch(self, batch_data:torch.Tensor):
        return (batch_data[:,:3,:,:].to(self.device), batch_data[:,3:,:,:].to(self.device))
    
    def run(self ): #每一个epoch测试30次
        self.net_G.train()
        self.net_D.train()
        f = open(self.logname, 'a+', buffering=1)
        batch_num, all_num = len(self.train_dataloader), len(self.train_dataloader.dataset)
        f.write("\n\n")
        for epoch in range(self.epoches):
            f.write(f"{self.remark} \n")
            f.write("*_"*30)
            f.write(f"时间：{datetime.datetime.now()} 总样本量：{all_num} 多少个batch: {batch_num} \n")

            for idx, batch_data in enumerate(tqdm(self.train_dataloader), 1):
                input_img, target_img = self.get_data_from_batch(batch_data) #范围超过+-1
                with autocast():
                    fake_img = self.net_G(input_img)
                    
                    #先更新D
                    self.optimizer_D.zero_grad()
                    predlabel_fake = self.net_D(fake_img.detach(), input_img)
                    predlabel_real = self.net_D(target_img, input_img)
                    realpatch_label, fakepatch_label = torch.ones_like(predlabel_fake, dtype=torch.float32, device=self.device), \
                        torch.zeros_like(predlabel_fake, dtype=torch.float32, device=self.device)
                    D_loss_fake, D_loss_real = self.gan_patchloss(predlabel_fake, fakepatch_label), \
                        self.gan_patchloss(predlabel_real, realpatch_label)
                    D_loss = D_loss_fake + D_loss_real

                    self.scaler.scale(D_loss).backward()
                    self.scaler.step(self.optimizer_D)
                    self.scaler.update()
                    self.scheduler_D.step(epoch + idx / batch_num)

                    #再更新G
                    self.optimizer_G.zero_grad()
                    predlabel_fake = self.net_D(fake_img, input_img)
                    G_loss_fake = self.gan_patchloss(predlabel_fake, realpatch_label)
                    G_loss_L1 = self.rec_loss(fake_img, target_img)
                    G_loss = G_loss_fake + G_loss_L1 #更改系数
                    if self.use_C_Loss == True:
                        C_Loss = diff_contrast_loss(fake_img, target_img, input_img[:,0:1,:,:], k = 10, use_diff = True, use_patch=True)
                        G_loss += C_Loss
                    else:
                        C_Loss = torch.tensor(0)
                    if self.use_SSIM_Loss == True:
                        SSIM_torch = torch_ssim(fake_img, target_img)
                        G_loss += 1 - SSIM_torch
                    else:
                        SSIM_torch = torch.tensor(0)

                    self.scaler.scale(G_loss).backward()
                    self.scaler.step(self.optimizer_G)
                    self.scaler.update()
                    self.scheduler_G.step(epoch + idx / batch_num)
                    self.G_rec_loss_list.append(G_loss_L1.cpu().detach().numpy())
                
                if len(self.G_rec_loss_list) % 50==0:
                    f.write(f"Epoch: {epoch}, Batch_idx: {idx}, lr: {self.optimizer_D.state_dict()['param_groups'][0]['lr']}, G_loss: {G_loss.cpu().item()}, D_loss: {D_loss.cpu().item()}, G_loss_L1: {G_loss_L1.cpu().item()} \n")
                    f.write(f"  G_loss_fake: {G_loss_fake.cpu().item()} \n")
                    f.write(f"  C_Loss: {C_Loss.cpu().item()} SSIM_torch: {SSIM_torch.cpu().item()} \n")
                if len(self.G_rec_loss_list) % 150 ==0:
                    self.draw_on_test(self.picture_savename, epoch, idx)
            
            if self.savename_GD != None:
                self.Test_model_metrics()
                self.save()
        
    
    @torch.no_grad()    
    def draw_on_test(self, name, epoch, idx, num=5):
        self.net_G.eval()
        input_img, target_img = self.get_data_from_batch(next(iter(self.test_dataloader)))
        batch_size, img_W = target_img.shape[0], target_img.shape[2]
        if num>batch_size:
            num = batch_size-1
        random_index = random.sample([i for i in range(batch_size)], num)
        input_img, target_img = input_img[random_index], target_img[random_index]
        fake_img = self.net_G(input_img).reshape(-1, img_W).detach().cpu()
        input_img = input_img.transpose(1,2).reshape(-1, 3*img_W).detach().cpu()
        target_img = target_img.reshape(-1, img_W).detach().cpu()
        target_fake_img = torch.concat([input_img, target_img, fake_img], dim=1)
        plt.figure(figsize=(10,10), dpi=250)
        plt.imshow(target_fake_img, cmap= 'gray')
        if self.picture_save != None:
            if not os.path.exists(self.picture_save):
                os.makedirs(self.picture_save) 
            plt.savefig(f"{self.picture_save}{name}_epoch_{epoch}_batch_{idx}.png")
        else:
            plt.show() #竖着两排 第一排真 第二排生成
        plt.close()
        self.net_G.train()

    def cal_metric(self, fake_img, target_img):
        #(bz, channel=1, H, W) np.array
        mse_list = [mse(target_img[i,0], fake_img[i,0]) for i in range(target_img.shape[0])]
        ssim_list =[ssim(target_img[i,0], fake_img[i,0], data_range=2.0) for i in range(target_img.shape[0])]
        psnr_list = [psnr(target_img[i,0], fake_img[i,0], data_range=2.0) for i in range(target_img.shape[0])]
        nrmse_list = [nrmse(target_img[i,0], fake_img[i,0]) for i in range(target_img.shape[0]) ]
        return  mse_list, ssim_list, psnr_list,  nrmse_list    
    
    @torch.no_grad()
    def Test_model_metrics(self,): #
        f = open(self.logname, 'a+', buffering=1)
        MSE_imgs, SSIM_imgs, PSNR_imgs, NRMSE_imgs = [], [], [], []
        self.net_G.eval()
        # 如果测试集太多，非常浪费时间，因此只测1024张图片
        for idx, batch_data in enumerate(tqdm(self.test_dataloader), 1):
            input_img, target_img = self.get_data_from_batch(batch_data)
            fake_img = self.net_G(input_img).detach().cpu().numpy()
            mse_list, ssim_list, psnr_list,  nrmse_list = self.cal_metric(fake_img, target_img.detach().cpu().numpy())
            MSE_imgs.extend(mse_list)
            SSIM_imgs.extend(ssim_list)
            PSNR_imgs.extend(psnr_list)
            NRMSE_imgs.extend(nrmse_list)
            if len(MSE_imgs) > 127:
                break
        # 要不要加方差
        MSE_imgs, SSIM_imgs, PSNR_imgs, NRMSE_imgs = np.mean(MSE_imgs), np.mean(SSIM_imgs), np.mean(PSNR_imgs), np.mean(NRMSE_imgs)
        self.net_G.train()
        f.write(f"MSE_imgs: {MSE_imgs}, SSIM_imgs: {SSIM_imgs}, PSNR_imgs: {PSNR_imgs}, NRMSE_imgs: {NRMSE_imgs} \n")

    def save(self,):
        checkpoint = {
            'net_G': self.net_G,
            'net_D': self.net_D,
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'scaler': self.scaler.state_dict()
        }

        torch.save(checkpoint, self.savename_GD)

# 当pix的数量很低时，如1x1,2x2，把像素当成词做注意力没有意义，所以在pix2pix没必要开
device = 'cuda:3'
config = {
        'image_size':256,
        'in_channels':3,
        'model_channels':64, #32没有64好
        'out_channels':1,
        'dropout':0.1,
        'conv_resample':False, #降采样采用池化而不是卷积，如果用num_res_blocks才考虑开启，作用于上下采样
        'resblock_updown':False, #用resnet降维升维
        'use_conv' : False, #不用卷积，如果用resblock_updown才考虑开启
        'num_res_blocks':1, #与resblock_updown互斥
        'channel_mult':(1, 2, 4, 4, 4, 4, 8, 8),#256。 128, 64, 32, 16, 8, 4，2,1(1, 2, 4, 4, 4, 4, 8, 8)
        # 'attention_resolutions':[1024], #关闭所有attn
        'num_head_channels':64,
        # 'use_new_attention_order': False,
        'use_tanh':True # maxmin归一化必须开启
}
net_G = UNetModel(**config).to(device)
net_D = SimpleDiscriminator(in_channels=config['in_channels']+config['out_channels']).to(device) #通道数合并


batch_size = 64
num_workers = 16
spinal_train_dataloader = DataLoader(spinal_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
spinal_test_dataloader = DataLoader(spinal_test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
args_dict = {
    'net_G':net_G,
    'net_D':net_D,
    'train_dataloader':spinal_train_dataloader,
    'test_dataloader':spinal_test_dataloader,
    'epoches':500, 
    'learning_rate':1e-4, 
    'use_C_Loss':False, #极限只用L1,SSIM，不用CLOSS
    'use_SSIM_Loss':True,

    'device':device, 
    'logname':"/home/zhangsenpeng/MRI_sequence_synthesis/log/1227_bigunet_pix2pix_only_spinal_baseline.txt", 
    'savename_GD':"/home/zhangsenpeng/MRI_sequence_synthesis/model/model_save/1227_bigunet_spinal_only_pix2pix_baseline_500epoch.pth", 
    'remark': "noCloss hasSSIMloss only spinal 500epoch MAEloss NoATTN!!!!",
    'picture_save': "/home/zhangsenpeng/MRI_sequence_synthesis/pictures/1227_bigunet_pix2pix_only_spinal_baseline/"
}
# 不太好
pix2pix_trianer = Trainer(**args_dict)
pix2pix_trianer.run()
