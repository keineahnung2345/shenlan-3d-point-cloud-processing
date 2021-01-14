# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 16:10:47 2021

@author: mimif
"""

import os
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm
import warnings

data_dir = "D:/data_object_box/"

"""
according to object3d.py's cls_type_to_id:
it will be {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4},
other classes will be -1
"""
classnames = ['Other', 'Vehicle', 'Pedestrian', 'Cyclist']
classes = ['Car', 'Pedestrian', 'Cyclist']
classmap = {-1:0,1:1,2:2,3:3,4:1}

class2samples = defaultdict(lambda : [])

for fname in os.listdir(data_dir):
    class2samples[classmap[int(fname[:-4].rsplit('_')[-1])]].append(fname)

class2sizes = defaultdict(lambda : [])
class2trainval = defaultdict(lambda : [[],[]])

for _class in range(4):
    print("class", _class)
    samples = class2samples[_class]
    
    for sample in tqdm(samples):
        fname = data_dir + sample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cloud = np.genfromtxt(fname, delimiter=' ', dtype=np.float32)
        # print(cloud.shape[0])
        class2sizes[_class].append(cloud.shape[0])
    
    # print(class2sizes[_class])
    
    cloud_size_thres = 1000
    selected = np.asarray(class2sizes[_class])>=cloud_size_thres
    samples = np.asarray(samples)[selected]
    class2samples[_class] = samples

# create a balanced validation set
val_size = int(min(map(len, class2samples.values())) * 0.2)

for _class in range(4):
    print("class", _class)
    samples = class2samples[_class]
    full_size = len(samples)
    print(_class, len(class2sizes[_class]), full_size)
    
    #https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    train_size = full_size - val_size
    
    random.shuffle(samples)
    
    samples = ["data_object_box/"+sample[:-4]+"\n" for sample in samples]
    
    with open("kitti_train.txt", "a") as f:
        f.writelines(samples[:train_size])
    
    with open("kitti_val.txt", "a") as f:
        f.writelines(samples[train_size:])
    
    

