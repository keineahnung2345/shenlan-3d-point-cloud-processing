# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:34:41 2020

@author: mimif
"""

from glob import glob
import numpy as np


modelnet40_dir = "D:\modelnet40_normal_resampled"

fnames = glob(modelnet40_dir + "\*\*.txt")

fnames = [fname for fname in fnames if '0' in fname 
          and '_' in fname.split('\\')[-1]]

fname = fnames[0]
fname = "C:\\Users\\mimif\\source\\repos\\ISS\\ISS\\airplane_0001.txt"

with open(fname, "r") as f:
    lines = f.readlines()

header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z normal_x normal_y normal_z
SIZE 4 4 4 4 4 4
TYPE F F F F F F
COUNT 1 1 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii

"""

with open(fname.replace(".txt", ".pcd"), "w") as f:
    cnt = len(lines)
    f.write(header.format(cnt, cnt))
    lines = [line.replace(',', ' ') for line in lines]
    f.writelines(lines)
    
