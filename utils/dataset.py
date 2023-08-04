import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import  random_split
from sklearn.model_selection import train_test_split
def acquire_all_patient(dir):
    all_patients = []
    all_group = os.listdir(dir)
    for group in all_group:
        group_path = os.path.join(dir, group)
        group_patients = os.listdir(group_path)
        for patient in group_patients:
            patient_path = os.path.join(group_path, patient)
            all_patients.append(patient_path)
    return all_patients

def acquire_all_img(temp_list):
    full_img_path=[]
    full_mask_path=[]
    for patient in temp_list:

        img_path=os.path.join(patient,'img')
        all_img=os.listdir(img_path)
        all_img=[os.path.join(img_path,i) for i in all_img]
        all_mask=[i.replace('img','mask') for i in all_img]

        full_img_path.extend(all_img)
        full_mask_path.extend(all_mask)

    return full_img_path,full_mask_path

class BasicDataset(Dataset):
    def __init__(self, img_path,mask_path,transform,scale=1,training=False):
        # self.imgs_dir = imgs_dir
        # self.masks_dir = masks_dir
        self.img_path=img_path
        self.mask_path=mask_path
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.training=training
        self.transform=transform
        self.ids_img=[i for i in range(len(self.img_path))]


    def __len__(self):
        return len(self.ids_img)
    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, idx):
        # idx = self.ids_img
        mask_file = self.mask_path[idx]
        img_file = self.img_path[idx]


        assert img_file.replace('img','mask')==mask_file


        mask = Image.open(mask_file).convert('L')
        img = Image.open(img_file).convert('RGB')


        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)


        if self.training:
            return (img, mask)
        return (img, mask, self.img_path[idx])




