# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 20:51:47 2021

@author: mimif
"""

import numpy as np
import os
import pcl
from tqdm import tqdm
import warnings

data_dir = "D:/data_object_box/"
normal_resampled_data_dir = "D:/data_object_box_normal_resampled/"

def normalize(arr):
    """
    normalize to [-0.5, 0.5]
    """
    arr = np.reshape(arr, (-1,3))
    lower = np.min(arr, axis=0)
    upper = np.max(arr, axis=0)
    center = (lower+upper)/2.0
    # move to (0,0,0)
    arr = arr - center
    # resize to (-0.5, 0.5)
    ratio = 1.0/(upper - lower).max()
    arr = arr * ratio
    return arr
    
"""
resample to 1000
"""

def downsampleCloud(cloud, leaf_size):
    vg = cloud.make_voxel_grid_filter()
    vg.set_leaf_size(leaf_size, leaf_size, leaf_size)
    cloud_filtered = vg.filter()
    return cloud_filtered

def downsample(arr, tgt_size, leaf_size):
    if arr.shape[0] >= tgt_size*1.5:
        if arr.shape[0] <= tgt_size*3:
            leaf_size = 0.02
        else:
            leaf_size = 0.05
        
        cloud = pcl.PointCloud()
        cloud.from_array(arr)
        
        cloud_down = downsampleCloud(cloud, leaf_size)
        
        if cloud_down.size >= tgt_size:
            arr_down = cloud_down.to_array()
            
        # print(arr.shape[0],"->",arr_down.shape[0])
    
    arr_final = arr[np.random.choice(arr.shape[0], tgt_size, replace=False), :]
    
    return arr_final
    
tgt_size = 1000
leaf_size = 0.02

# for fname in os.listdir(data_dir):
for fname in tqdm(os.listdir(data_dir)):
    with warnings.catch_warnings(): # to disable "empty file" warning
        warnings.simplefilter("ignore")
        arr = np.genfromtxt(data_dir + fname, dtype=np.float32)
        # arr = pcl.np.loadtxt(data_dir + fname)
    
    if arr.shape[0] < tgt_size: continue
    
    arr_norm = normalize(arr)
    # print(np.min(arr, axis=0), np.max(arr, axis=0))
    # print(np.min(arr_norm, axis=0), np.max(arr_norm, axis=0))
    
    arr_norm_down = downsample(arr_norm, tgt_size, leaf_size)
    
    np.savetxt(normal_resampled_data_dir+fname, arr_norm_down)
    # break
