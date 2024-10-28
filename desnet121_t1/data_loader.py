# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:14:38 2024

@author: pky0507
"""

import os
import numpy as np
import pandas as pd

def get_data_list(root='/dataset/IPMN_Classification/', t = 1, center = None):
    image_list = []
    label_list = []
    center_names = [['nyu'], ['CAD', 'MCF'], ['northwestern', 'NU'], ['AHN', 'ahn'], ['mca'], ['IU'], ['EMC']]
    
    df = pd.read_excel(os.path.join(root, 'IPMN_labels_t'+str(t)+'.xlsx'), usecols=[0, 3])
    df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN
    names = df_cleaned.iloc[:, 0].values
    labels = df_cleaned.iloc[:, 1].to_numpy(dtype=np.int64)//2 # we treat no/low-risk as 0 and high-risk as 1
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
                label_list.append(labels[i])
                break
    return image_list, label_list

def split_data_list(image:list, label:list, ratio=0.8, seed=None):
    rng = np.random.default_rng(seed)
    classes = set(label)
    train_ind = []
    test_ind = []
    for j in classes:
        ind = [i for i, x in enumerate(label) if x == j]
        split = int(len(ind) * ratio)
        rng.shuffle(ind)
        train_ind += ind[:split]
        test_ind += ind[split:]
    rng.shuffle(train_ind)
    rng.shuffle(test_ind)
    train_image = [image[i] for i in train_ind]
    train_label = [label[i] for i in train_ind]
    test_image = [image[i] for i in test_ind]
    test_label = [label[i] for i in test_ind]    
    return train_image, train_label, test_image, test_label