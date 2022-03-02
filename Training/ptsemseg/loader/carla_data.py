from __future__ import print_function
import torch
from torch.utils import data
DEBUG = False
import numpy as np
import sys
import random
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
import os

# this script is the dataloader for Carla
class RRC_video(transforms.RandomResizedCrop):
    def __call__(self, imgs, imgs2, heatmap, mode, jitter, affine_tr, type_carla):
        
        for im in range(1, len(imgs)):
            assert imgs[im].size() == imgs[0].size()
            assert imgs2[im].size() == imgs2[0].size()
        
        to_img = transforms.ToPILImage()

        rand  = random.random()
        i=0
        j=0
        h=0
        w=0
        
#        angle = [random.uniform(-6, 6) for x in range(4)] # used to camera instability
#        max_dx = 0.1 * 512
#        max_dy = 0.1 * 512
#        translations = [(np.round(random.uniform(-max_dx, max_dx)),
#                        np.round(random.uniform(-max_dy, max_dy))) for x in range(4)]
    
        
        if heatmap is not None :
            heatmap = to_img(heatmap)
            if mode == "train":
                heatmap = TF.center_crop(heatmap, 512) # keep the center of an image to accelerate training
#            heatmap = TF.affine(heatmap, *affine_tr, resample=0, fillcolor=0)
#            heatmap = TF.affine(heatmap, angle[3], translations[3], 1.0, 0.0, resample=0, fillcolor=0)
            if mode == "train":
                i, j, h, w = self.get_params(heatmap, self.scale, self.ratio)  
                if rand > 0.5:
                    heatmap = TF.hflip(heatmap) # flip horizontally
    
                heatmap = TF.resized_crop(heatmap, i, j, h, w, self.size, Image.NEAREST)
                heatmap = TF.to_tensor(heatmap)
        else:
            i, j, h, w = self.get_params(to_img(imgs[0]), self.scale, self.ratio)  
                
        
        for imgCount in range(len(imgs)):
            img = to_img(imgs[imgCount])
#            if mode == "train":
            img = TF.center_crop(img, 512)
#           img = TF.affine(img, *affine_tr, resample=2, fillcolor=(0,0,0))
#           img = TF.affine(img, angle[imgCount], translations[imgCount], 1.0, 0.0, resample=2, fillcolor=(0,0,0))
            labels = to_img(imgs2[imgCount])
#            if mode == "train":
            labels = TF.center_crop(labels, 512)            
#           labels = TF.affine(labels, *affine_tr, resample=0, fillcolor=254)
#           labels = TF.affine(labels, angle[imgCount], translations[imgCount], 1.0, 0.0, resample=0, fillcolor=254)
            if mode == "train":
                if rand > 0.5:
                    img = TF.hflip(img)
                    labels = TF.hflip(labels)
                img = jitter(img) # aplly color modifier
            img = TF.to_tensor(img)
            
            imgs2[imgCount] = torch.from_numpy(np.array(labels)).long()
            
            if type_carla ==1 :
                imgs[imgCount] = TF.normalize(img, mean=(0.5188, 0.5181, 0.5052), std=(0.2478, 0.2513, 0.2565)) # normalize image
            else :
                imgs[imgCount] = TF.normalize(img, mean=(-0.1096, -0.0941, -0.1247), std=(0.8866, 0.8912, 0.9073))#mean=(-0.0144, -0.0510, -0.0723), std=(0.9829, 1.0020, 1.0121)
        

        return imgs, imgs2, heatmap
    
class Carla(data.Dataset):

    def __init__(self, root_dir, mode='train', time_step=4, transform=True, resize=None,type_carla=1, with_heatmap=False):
        self.mode = mode
        self.root_dir=root_dir
        self.time_step=time_step
        self.transform=transform
        self.resize = resize
        self.type=type_carla
        self.with_heatmap = with_heatmap
        self.n_classes=13
        
        if self.mode.lower() == 'train':  # different part of the dataset to choose
          self.data_dir = os.path.join(root_dir,'train')
          self.labels_dir = os.path.join(root_dir,'trainannot')
          self.heatmap_dir = os.path.join(root_dir,'trainheat')
        elif self.mode.lower() == 'val':
          self.data_dir = os.path.join(root_dir,'val')
          self.labels_dir = os.path.join(root_dir,'valannot')
          self.heatmap_dir = os.path.join(root_dir,'valheat')
        elif self.mode.lower() == 'test':
          self.data_dir = os.path.join(root_dir,'test')
          self.labels_dir = os.path.join(root_dir,'testannot')
          self.heatmap_dir = os.path.join(root_dir,'testheat')
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
        self.length = len(next(os.walk(self.data_dir))[2])
    def transforms(self, img, labels, heat):

        return img ,labels, heat


    def __getitem__(self, index):
        index = index * 4 # ge tthe index of the next sequence
        data_path = []
        labels_path = []
        imgs_data = []
        imgs_labels = []
        heatmap = None
        
        self.jitter_transform = transforms.ColorJitter.get_params((0.5,1.5),(0.5,1.5),(0.5,1.5),None) # modify colors on the image
        self.affine_transform = transforms.RandomAffine.get_params((-15,15),(0.15,0.1),None,None,(512,512))        #move the image to mimic camera instability
        for i in range(0,self.time_step):
            wzero=(str(index+i)).zfill(5)
            data_path.append( os.path.join(self.data_dir,wzero+".png"))
            labels_path.append(os.path.join(self.labels_dir,wzero+".png"))
            if i == (self.time_step - 1):
                if self.with_heatmap :
                    
                    heatmap = cv2.imread(os.path.join(self.heatmap_dir,wzero+".png"),0)
                
          
        for i in range(0,self.time_step):
            img = cv2.imread(data_path[i])
            labels = cv2.imread(labels_path[i],0)
            
            if self.mode =="val" : 
                if self.transform is not None:
                
                    if self.resize is not None:
                        img = cv2.resize(img,self.resize)
                        labels = cv2.resize(labels,self.resize,interpolation=cv2.INTER_NEAREST)
                        if self.with_heatmap :
                            heatmap = cv2.resize(heatmap,self.resize,interpolation=cv2.INTER_NEAREST)
                    
            img = np.moveaxis(np.array(img),-1,0)
            labels[labels == 0]= 13
            labels = labels -1
            img = torch.from_numpy(img).float()  /255.0
            labels = torch.from_numpy(np.array(labels))
            imgs_data.append(img)
            imgs_labels.append(labels)
             
        if self.with_heatmap : # heatmap are used with attentive cutmix that is not used anymore
                heatmap = cv2.resize(heatmap,self.resize,interpolation=cv2.INTER_NEAREST)
                heatmap = torch.from_numpy(heatmap).float()  /255.0 
        if self.transform : # apply augmentations
            imgs_data, imgs_labels, heatmap = RRC_video(None, scale=(0.7, 1.0),
                                               ratio=(5.0/6.0,6.0/5.0))(imgs_data, imgs_labels,
                                                     heatmap, self.mode, self.jitter_transform, 
                                                     self.affine_transform, self.type)   
#            print('INSIDE CARLA DATALOADER')
#            print(type(imgs_data))
#            print(type(imgs_data[0]))
#            print(type(imgs_labels))
#        imgs_data = torch.stack(imgs_data,dim=0)
        imgs_labels = torch.stack(imgs_labels,dim=0)

        if self.with_heatmap :
            return imgs_data, imgs_labels[-1].long(), heatmap
        else:
            return imgs_data, imgs_labels[-1].long()


    def __len__(self):
       return self.length // 4
