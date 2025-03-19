import nibabel as nib
import os
import numpy as np
import random
from PIL import Image
from torch.utils import data
import torch
from sklearn.utils import shuffle
from config import dataset_folder
###
class Brats17(data.Dataset):
    def __init__(self, root, transforms = None, train = True, test = False, val = False):
        self.test = test
        self.train = train
        self.val = val
        if self.train:
            self.root = '/tmp/code/brain_seg/data00/train/00'
            self.folderlist = os.listdir(self.root)
        elif self.val:
            self.root = ''
        elif self.test:
            self.root = ''
            self.folderlist = os.listdir(os.path.join(self.root))
    def __getitem__(self,index):
        if self.train:                            
            if 1 > 0 :
                ss = 64 
                sss = 96
                #print(self.folderlist[index])
                path = self.root
                img = np.load(os.path.join(path,self.folderlist[index]))
                img = np.asarray(img)
                #print(img.shape)
                # C * W * H
                #处理和剪裁
                index_x = np.random.randint(ss,img.shape[1]-ss,size=1)       #ss和sss分别是x和y方向上的半剪裁尺寸
                index_y = np.random.randint(sss,img.shape[2]-sss,size=1)     #分别随机生成x和y坐标上的索引     
                img_in = img[:,index_x[0]-ss:index_x[0]+ss,index_y[0]-sss:index_y[0]+sss]   
                #print("img_in", img_in)
                img_out = img_in[0:1,:,:].astype(float)            #0
                label_out = img_in[1,:,:].astype(float)            #1
                ##二分类问题
                label_out[label_out != int(dataset_folder)] = 0
                label_out[label_out == int(dataset_folder)] = 1
                #print(img_in.shape)
                img = torch.from_numpy(img_out).float()            #将其转换为PyTorch张量
                label = torch.from_numpy(label_out).long()
                #print("success")
                #print(label_out.max(),label_out.min())  
        elif self.val:
            path = self.root       
            img = np.load(os.path.join(path,self.folderlist[index]))
            img = np.asarray(img)
            img_out = img[0,:,:,:].astype(float)
            label_out = img[1,:,:,:].astype(float)
            #print(img.shape)
            img = torch.from_numpy(img_out).unsqueeze(0).float()     
            label = torch.from_numpy(label_out).long()
        else:
            print('###$$$$$$$$$$$$$$$$$$$^^^^^^^^^^^^^')     
        return img, label
    def __len__(self):
        return len(self.folderlist)







