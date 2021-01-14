# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:16:05 2021

@author: mimif
"""

import os
import numpy as np
from PointNet import PointNetSmall, evaluate_one
import torch
from torch import nn
from glob import glob
import operator
from kitti_utils import get_calib, get_image_shape, save_kitti_format
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA

cluster_dir = "D:/data_cluster/"
calib_dir = "D:/data_object_calib/training/calib"
image_dir = "D:/data_object_image_2/training/image_2/"
result_dir = "result/data/"
cluster_rect_dir = "D:/PointCloudCourse/HW10/result/rect_clusters/"
model_path = "small/mypn_2000.pth"
CLASSES = ['Other', 'Vehicle', 'Pedestrian', 'Cyclist']
truncated = 0.0
occluded = 0
alpha = 0.0
device = torch.device('cpu')
cloud_size_thres = 1000
allow_rotated = False

def check_valid(cloud):
    # if cloud.shape[0] < cloud_size_thres:
    #     return False
    extent = np.max(cloud,axis=0)-np.min(cloud,axis=0)
    if np.max(extent) > 3:
        return False
    return True
    
def evaluate_classification():
    pass

pn = PointNetSmall(4, pool_size=cloud_size_thres, dropout_rate=0)
pn.load_state_dict(torch.load(model_path,
                              map_location=device))
pn.eval()

for sample_id in tqdm(range(10,90)):
    sample_fnames = glob(cluster_dir + "{:06d}*".format(sample_id))
    calib = get_calib(calib_dir, sample_id)
    image_shape  = get_image_shape(image_dir, sample_id)
    
    boxes3d = []
    classes = []
    scores = []
    
    for sample_fname in sample_fnames:
        cluster_id = sample_fname.rsplit('/',1)[-1].split('.')[0].split('_')[-1]
        pts_lidar = np.loadtxt(sample_fname, delimiter=' ')
        
        """
        calibration: lidar's coord -> rect's coord
        """
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
        # pts_rect = pts_lidar[:, 0:3]
        np.savetxt(cluster_rect_dir + "{:06d}_{}.txt".format(sample_id, cluster_id), pts_rect)
        
        if not check_valid(pts_rect): continue
        
        # return 0, 1, 2 or -1
        pn.pool = nn.MaxPool1d(pts_rect.shape[0])
        classid, score = evaluate_one(pn, pts_rect)
        if classid == 0: continue
        _class = "Car" #CLASSES[classid]
        classes.append(_class)
        scores.append(score)
        
        location = (np.max(pts_rect, axis=0)+np.min(pts_rect, axis=0))/2
        
        if not allow_rotated:
            dimensions = np.max(pts_rect, axis=0)-np.min(pts_rect, axis=0)
            # (h,w,l) <-> (y,z,x)
            dimensions = dimensions[[1,2,0]]
            rotation_y = 0.0
        else:
            dimensions = np.max(pts_rect, axis=0)-np.min(pts_rect, axis=0)
            y_dimensions = dimensions[1]
            
            #project onto xz plane
            pts_rect_plane = pts_rect[:,[0,2]]
            pca = PCA(n_components=2)
            pca.fit(pts_rect_plane)
            
            #The components are sorted by explained_variance_
            directions = pca.components_
            ratios = pca.explained_variance_ratio_
            
            major_direction, minor_direction = directions
            
            #major_direction[0]: major_direction.dot(x_axis)
            rotation_y = np.arccos(major_direction[0])
            if major_direction[0] < 0: rotation_y *= -1
            
            pts_rect_plane_new = np.hstack(
                [pts_rect_plane.dot(major_direction)[...,np.newaxis],
                pts_rect_plane.dot(minor_direction)[...,np.newaxis]]
                )
            x_dimensions, z_dimensions = \
                np.max(pts_rect_plane_new, axis=0)-np.min(pts_rect_plane_new, axis=0)
            
            dimensions = np.hstack([y_dimensions, z_dimensions, x_dimensions])
        
        boxes3d.append(location.tolist()+dimensions.tolist()+[rotation_y])
    
    boxes3d = np.asarray(boxes3d)
    scores = np.asarray(scores)
    save_kitti_format(classes, sample_id, calib, boxes3d, result_dir, scores,
                          image_shape)