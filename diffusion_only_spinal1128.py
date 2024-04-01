import numpy as np
import matplotlib.pyplot as plt
from IPython.display import  clear_output
from IPython.display import display as dp
import seaborn as sns
import random
import os
import datetime
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from ema_pytorch import EMA

import sys 
sys.path.append("..") 
# from model.unet import * #这个很可以
from model.diffusion_big1126 import *
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
    def __init__(self, net_G:nn.Module, train_dataloader, test_dataloader, 
                 epoches, learning_rate, device, \
                logname, gradient_accumulate_every = 4, savename_G = None, remark = "Nothing", picture_save = None,
                ema_update_every = 10, ema_decay = 0.995, ):
        self.picture_savename = 'spinal_test'
        self.epoches = epoches
        self.lr = learning_rate
        self.device = device
        self.logname = logname
        self.savename_G = savename_G
        self.remark = remark
        self.picture_save = picture_save
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.net_G = net_G.to(device)
        self.ema = EMA(net_G, beta = ema_decay, update_every = ema_update_every).to(device)#设置指数滑动平均
        self.optimizer_G = optim.Adam(net_G.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-6)
        self.scheduler_G = CosineAnnealingWarmRestarts(self.optimizer_G, T_0=1, T_mult=1) #开1倍扩增
        self.scaler = GradScaler()
        self.loss_list = []
        self.gradient_accumulate_every = gradient_accumulate_every
        self.start_epoch = 0

    def get_data_from_batch(self, batch_data:torch.Tensor):
        return (batch_data[:,:3,:,:].to(self.device), batch_data[:,3:,:,:].to(self.device))
    
    def run(self ): #每一个epoch测试30次
        self.net_G.train()
         #混合精度浮点运算
        f = open(self.logname, 'a+', buffering=1)
        batch_num, all_num = len(self.train_dataloader), len(self.train_dataloader.dataset)
        f.write("\n\n")
        for epoch in range(self.start_epoch, self.epoches):
            f.write(f"{self.remark} \n")
            f.write("*_"*30)
            f.write(f"时间：{datetime.datetime.now()} 总样本量：{all_num} 多少个batch: {batch_num} \n")

            for idx, batch_data in enumerate(tqdm(self.train_dataloader, desc = 'Trainer running', total=batch_num), 1):
                input_img, target_img = self.get_data_from_batch(batch_data) #范围超过+-1
                with autocast():
                    loss = self.net_G(target_img, input_img) / self.gradient_accumulate_every #模型训练仍用netG的参数，模型保存和推理用指数平滑的参数
                    self.loss_list.append(loss.cpu().detach().numpy())
                    if idx % self.gradient_accumulate_every==0:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer_G)
                        self.scaler.update()
                        self.scheduler_G.step(epoch + idx / batch_num)
                        self.optimizer_G.zero_grad()
                        self.ema.update()
                    
                
                if len(self.loss_list) % 50==0:
                    self.save(epoch, loss)
                    f.write(f"Epoch: {epoch}, Batch_idx: {idx}, lr: {self.optimizer_G.state_dict()['param_groups'][0]['lr']}, loss: {loss.cpu().item()} \n")

                if len(self.loss_list) % 200 ==0:
                    self.draw_on_test(self.picture_savename, epoch, idx)
            
            if self.savename_G != None:
                self.save(epoch, loss)
                self.Test_model_metrics()
        
    
    @torch.no_grad()    
    def draw_on_test(self, name, epoch, idx, num=5):
        self.ema.ema_model.eval()
        input_img, target_img = self.get_data_from_batch(next(iter(self.test_dataloader)))
        batch_size, img_W = target_img.shape[0], target_img.shape[2]
        if num>batch_size:
            num = batch_size-1
        random_index = random.sample([i for i in range(batch_size)], num)
        input_img, target_img = input_img[random_index], target_img[random_index]

        fake_img = self.ema.ema_model.sample()(shape =(num, 1, 256, 256), source_img = input_img).reshape(-1, img_W).detach().cpu()

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
        self.ema.ema_model.eval()
        # 如果测试集太多，非常浪费时间，因此只测1024张图片
        for idx, batch_data in enumerate(tqdm(self.test_dataloader), 1):
            input_img, target_img = self.get_data_from_batch(batch_data)
            fake_img = self.ema.ema_model.sample()(shape =target_img.shape, source_img = input_img).detach().cpu().numpy()

            mse_list, ssim_list, psnr_list,  nrmse_list = self.cal_metric(fake_img, target_img.detach().cpu().numpy())
            MSE_imgs.extend(mse_list)
            SSIM_imgs.extend(ssim_list)
            PSNR_imgs.extend(psnr_list)
            NRMSE_imgs.extend(nrmse_list)
            if len(MSE_imgs) >= 47:
                break
        # 要不要加方差
        MSE_imgs, SSIM_imgs, PSNR_imgs, NRMSE_imgs = np.mean(MSE_imgs), np.mean(SSIM_imgs), np.mean(PSNR_imgs), np.mean(NRMSE_imgs)
        self.net_G.train()
        f.write(f"MSE_imgs: {MSE_imgs}, SSIM_imgs: {SSIM_imgs}, PSNR_imgs: {PSNR_imgs}, NRMSE_imgs: {NRMSE_imgs} \n")

    def save(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model': self.ema.ema_model,
            'optimizer': self.optimizer_G.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict(),
            'loss': loss,
        }

        torch.save(checkpoint, self.savename_G)
    
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.start_epoch = checkpoint['epoch']
        self.model = checkpoint['model'].to(self.device)
        self.ema.load_state_dict(checkpoint['ema'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer'])


device = 'cuda:0'
config = {
    'image_size':256,
    'in_channels':1+3,
        'model_channels':64,
        'out_channels':1,
        'num_res_blocks':2,
        'attention_resolutions':[8], #4
        'channel_mult':(1, 2, 4, 4), #(1, 2, 4 ,4)
        'use_fp16':'True',
        'num_head_channels':64,
        'legacy':False


}
diffusion_model = GaussianDiffusion(
    image_size=256,
    model = UNetModel(**config).to(device),
    timesteps = 1000,           # number of steps
    sampling_timesteps = 16,    # using ddim for faster inference 
    objective = 'pred_v', #pred_v pred_x0崩 pred_noise

).to(device)

batch_size = 32
num_workers = 8
spinal_train_dataloader = DataLoader(spinal_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
spinal_test_dataloader = DataLoader(spinal_test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
args_dict = {
    'net_G':diffusion_model,
    'train_dataloader':spinal_train_dataloader,
    'test_dataloader':spinal_test_dataloader,
    'epoches':500, 
    'learning_rate':1e-4, 
    'gradient_accumulate_every':1,
    'device':device, 
    'logname':"/home/zhangsenpeng/MRI_sequence_synthesis/log/1128_bigunet_pix2pix_only_spinal_diffusion_1000.txt", 
    'savename_G': "/home/zhangsenpeng/MRI_sequence_synthesis/model/model_save/1128_diffusion_model_opt_1000.pth",
    'remark': "1000 patients only spinal diffusion",
    'picture_save': "/home/zhangsenpeng/MRI_sequence_synthesis/pictures/1128_bigunet_pix2pix_only_spinal_diffusion_1000/"
}
# 不太好
pix2pix_trianer = Trainer(**args_dict)
pix2pix_trianer.run()