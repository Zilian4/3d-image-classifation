# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:14:38 2024

@author: pky0507
"""

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import monai
import pandas as pd

from monai.data import DataLoader, ImageDataset,decollate_batch
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    MixUp,
    CutMix
)
    
# class PanSeg(Dataset):
#     def __init__(self, root='/dataset/IPMN_Classification/', t = 1, center = None):
#         self.data = []
#         self.label = []
#         self.center_name = [['nyu'], ['CAD', 'MCF'], ['northwestern', 'NU'], ['AHN', 'ahn'], ['mca'], ['IU'], ['EMC']]
        
#         df = pd.read_excel(os.path.join(root, 'IPMN_labels_t'+str(t)+'.xlsx'), usecols=[0, 3])
#         df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN
#         names = df_cleaned.iloc[:, 0].values
#         labels = df_cleaned.iloc[:, 1].to_numpy(dtype=int)
#         if center == None:
#             center = np.arange(len(self.center_name))
#         elif isinstance(center, int):
#             center = [center]
#         center_name = []
#         for i in center:
#             center_name += self.center_name[i]
#         for i in range(len(names)):
#             name = names[i].replace('.nii.gz', '')
#             for c in center_name:
#                 if c in name:
#                     self.data.append(os.path.join(root, 't'+str(t)+'_clean_ROI', name+'.nii.gz'))
#                     self.label.append(labels[i])
#                     break


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image  = nib.load(self.data[idx]).get_fdata(dtype=np.float32)
#         label  = self.label[idx]

#         return image, label

# dataset = PanSeg(t = 1, center = None)
# for i in range(len(dataset)):
#     x, y = dataset.__getitem__(i)

def get_data_list(root='/dataset/IPMN_Classification/', t = 1, center = None):
    image_list = []
    label_list = []
    center_names = [['nyu'], ['CAD', 'MCF'], ['northwestern', 'NU'], ['AHN', 'ahn'], ['mca'], ['IU'], ['EMC']]
    
    df = pd.read_excel(os.path.join(root, 'IPMN_labels_t'+str(t)+'.xlsx'), usecols=[0, 3])
    df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN
    names = df_cleaned.iloc[:, 0].values
    labels = df_cleaned.iloc[:, 1].to_numpy(dtype=int)
    if center == None:
        center = np.arange(len(center_names))
    elif isinstance(center, int):
        center = [center]
    center_name = []
    for i in center:
        center_name += center_names[i]
    for i in range(len(names)):
        name = names[i].replace('.nii.gz', '')
        for c in center_name:
            if c in name:
                image_list.append(os.path.join(root, 't'+str(t)+'_clean_ROI', name+'.nii.gz'))
                label_list.append(labels[i].astype(np.int64))
                break
    return image_list, label_list

# image_list, label_list = get_data_list()

# train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
# test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
# train_ds = ImageDataset(image_files=image_list, labels=label_list, transform=train_transforms)