import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
# from torchvision.transforms import v2
import os
from pathlib import Path
from glob import glob
import random
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from IPython import display

class MRI_patient_Dataset(Dataset):
    def __init__(self, dir_path, focal_str = '*', size=256): #focal_str='schwannoma'
        self.fnames = glob(os.path.join(dir_path, focal_str))
        #应用特征增强
        self.transform = T.Compose([ 
        T.Resize(size, antialias=True),
        T.CenterCrop(size),       
        T.RandomHorizontalFlip(), # 随机水平，垂直翻转 其他的增强以后考虑
        T.RandomVerticalFlip(),
        T.ToTensor(), #(H, W, channels)--(channels, H, W),同时0--255除以255后变成0--1
        T.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)), # 原始数据[0,1]转换后每个通道[-1,+1]
        ])
    
    def __getitem__(self,idx):
        img = Image.open(self.fnames[idx])
        img = self.transform(img)
        return img
 
    def __len__(self):
        return len(self.fnames)

class MRI_patient_Dataset_fortest(Dataset):
    def __init__(self, dir_path, focal_str = '*', size=256):
        self.fnames = glob(os.path.join(dir_path, focal_str))
        #应用特征增强
        self.transform = T.Compose([ 
        T.Resize(size, antialias=True),
        T.CenterCrop(size),       
        T.ToTensor(), #(H, W, channels)--(channels, H, W),同时0--255除以255后变成0--1
        # diff关闭
        T.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)), # 原始数据[0,1]转换后每个通道[-1,+1]
        ])
    
    def __getitem__(self,idx):
        img = Image.open(self.fnames[idx])
        # print(f"当前文件路径是：{self.fnames[idx]}")
        img = self.transform(img)
        return img
 
    def __len__(self):
        return len(self.fnames)

class MRI_patient_Dataset_fortestpatient(Dataset):
    def __init__(self, dir_path, patient_id, size=256):
        self.fnames = glob(os.path.join(dir_path, '*'+patient_id+'*')) #注意：当截面超过10之后不是按照截面顺序
        #应用特征增强
        self.transform = T.Compose([ 
        T.Resize(size, antialias=True),
        T.CenterCrop(size),       
        T.ToTensor(), #(H, W, channels)--(channels, H, W),同时0--255除以255后变成0--1
        T.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)), # 原始数据[0,1]转换后每个通道[-1,+1]
        ])
    
    def __getitem__(self,idx):
        img = Image.open(self.fnames[idx])
        # print(f"当前文件路径是：{self.fnames[idx]}")
        img = self.transform(img)
        return img
 
    def __len__(self):
        return len(self.fnames)
# mean-std归一化效果不如maxmin
class MRI_patient_meanstd_Dataset(Dataset):
    def __init__(self, dir_path):
        self.fnames = glob(os.path.join(dir_path, '*'))
        self.transform = T.Compose([ 
        T.Resize(256, antialias=True),
        T.CenterCrop(256),       
        T.RandomHorizontalFlip(), # 随机水平，垂直翻转 其他的增强以后考虑
        T.RandomVerticalFlip(),
        T.ToTensor(), #(channels, H, W)
        ])
    
    def __getitem__(self,idx):
        img = Image.open(self.fnames[idx]) #输入8位png图片
        img = self.transform(img) #(channels, H, W)
        return self.norm(img) #self.norm(
 
    def __len__(self):
        return len(self.fnames)

    def norm(self, img:torch.Tensor):
        channels = img.shape[0]
        mean_one_channel = torch.mean(img.reshape(channels,-1), dim=-1).reshape(channels, 1, 1)
        std_one_channel = torch.std(img.reshape(channels,-1), dim=-1).reshape(channels, 1, 1)
        return (img - mean_one_channel)/(std_one_channel+1e-10)


def norm_layerHW(imgdata, method = '8bit'): #多层单通道图片

    layers = imgdata.shape[0]
    max_value = np.max(imgdata.reshape(layers,-1),axis=-1).reshape(layers,1,1)
    min_value = np.min(imgdata.reshape(layers,-1),axis=-1).reshape(layers,1,1)
    imgdata = (imgdata-min_value)/(max_value-min_value+1e-10) #0-1之间浮点数
    if method == '8bit':
        imgdata = (255 * imgdata).astype(np.uint8) #转成int8
    if method == '0-1':
        pass
    if method == '+-1':
        imgdata = 2*imgdata -1
    return imgdata

def save_layers_img(imgdata, main_dir, descri_str):
    imgdata = norm_layerHW(np.array(imgdata), method = '8bit')

    os.makedirs(main_dir, exist_ok=True)
    for i in range(imgdata.shape[0]):
        img = Image.fromarray(imgdata[i,:,:])
        i_savepath = os.path.join(main_dir, descri_str+'_'+str(i)+'.png')
        img.save(i_savepath)


# 显示图片
def show_slices_2D(brain_tumor_img_data, merge=True, dpi = 300, figsize = (8,6), savepath = None):
    brain_tumor_img_data = np.array(brain_tumor_img_data)
    channels = brain_tumor_img_data.shape[0]
    if merge:
      plt.figure(figsize=figsize,dpi=dpi)
      plt.imshow(brain_tumor_img_data.transpose(1,2,0),  origin="lower")
      pass
    else:
      plt.figure(figsize=figsize,dpi=dpi)
      for ch in range(channels):
        plt.subplot(1, channels,ch+1)
        plt.imshow(brain_tumor_img_data[ch,:,:], cmap="gray", origin="upper") #lower
    if savepath !=None:
        plt.savefig(savepath)
    plt.show()
    plt.close()
  
#动态图
def show_slices_2D_with_slider(img_data: np.array, merge=True, time = 0.15):
    print(f"img_data shape: {img_data.shape}")
    channels, layers, _, _ = img_data.shape

    for idx in range(layers):
        if merge:
            plt.subplots(figsize=(4, 4), dpi=300)
            plt.subplots_adjust(bottom=0.25)  # Adjust the bottom to make space for the slider
            plt.cla()
            plt.title(f"{idx}")
            plt.imshow(img_data[:, idx, :, :].transpose(1,2,0), cmap='gray', origin='lower') #(3,266,256)——(256,256,3)
        else:
            plt.subplots(figsize=(6, 6), dpi=300)
            for ch in range(channels):
                plt.subplot(1, channels, ch+1) #warning
                plt.cla()
                plt.title(f"{idx}")
                plt.imshow(img_data[ch,idx,:, :], cmap="gray", origin="lower")
        display.clear_output(wait=True)
        plt.pause(time)
        
    plt.show(block=True)
    plt.close()

# 显示像素值分布
def pix_values_distribution(image_matrix, dpi = 200, figsize = (3,2)):
    flat_values = image_matrix.flatten()
    # 绘制直方图
    plt.figure(figsize=figsize,dpi=dpi)
    plt.hist(flat_values, bins=50, color='blue', alpha=0.7)
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    plt.close()