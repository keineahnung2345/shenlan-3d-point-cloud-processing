# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:34:41 2020

@author: mimif
"""

from glob import glob
import numpy as np
import struct

def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    # return np.asarray(pc_list, dtype=np.float32).T
    return np.asarray(pc_list, dtype=np.float32)

input_dir = "D:\\PointCloudCourse\\HW9\\registration_dataset\\point_clouds\\"
output_dir = "D:\\PointCloudCourse\\HW9\\registration_dataset\\pcd\\"

input_type = "bin"
fnames = glob(input_dir + "*." + input_type)

for fname in fnames:
    if input_type == "txt":
        with open(fname, "r") as f:
            lines = f.readlines()
    elif input_type == "bin":
        lines = read_velodyne_bin(fname)
        lines = [','.join(
            [str(token) for token in line]
            )+"\n" for line in lines]
    
    if input_type == "txt":
        s1 = "normal_x normal_y normal_z"
        s2 = "4 4 4"
        s3 = "F F F"
        s4 = "1 1 1"
    elif input_type == "bin":
        s1 = s2 = s3 = s4 = ""
    
    header = """# .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z {}
    SIZE 4 4 4 {}
    TYPE F F F {}
    COUNT 1 1 1 {}
    WIDTH {}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS {}
    DATA ascii
    """
    
    # fnames = [fname for fname in fnames if '0' in fname 
    #           and '_' in fname.split('\\')[-1]]
    
    fname = output_dir + fname.rsplit('\\', 1)[-1].split(".")[0] + ".pcd"
    
    
    with open(fname, "w") as f:
        cnt = len(lines)
        f.write(header.format(s1, s2, s3, s4, cnt, cnt))
        lines = [line.replace(',', ' ') for line in lines]
        f.writelines(lines)
    
