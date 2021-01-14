# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 17:09:05 2020

@author: mimif
"""

#https://github.com/sshaoshuai/PointRCNN/issues/148

import numpy as np
import mayavi
import mayavi.mlab as mlab
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull
import scipy
from tqdm import tqdm
import struct
import os
from PIL import Image
from matplotlib.path import Path
from kitti_utils import *

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

def read_oxford_bin(bin_path):
    '''
    :param path:
    :return: [x,y,z,nx,ny,nz]: 6xN
    '''
    data_np = np.fromfile(bin_path, dtype=np.float32)
    return np.transpose(np.reshape(data_np, (int(data_np.shape[0]/6), 6)))



def draw_lidar(pc, color=None, fig=None, bgcolor=(0,0,0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=pts_color, mode=pts_mode, colormap = 'gnuplot', scale_factor=pts_scale, figure=fig)

    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)

    # draw fov (todo: update to real sensor spec.)
    fov=np.array([  # 45 degree
        [20., 20., 0.,0.],
        [20.,-20., 0.,0.],
    ],dtype=np.float64)

    mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    # draw square region
    TOP_Y_MIN=-20
    TOP_Y_MAX=20
    TOP_X_MIN=0
    TOP_X_MAX=40
    TOP_Z_MIN=-2.0
    TOP_Z_MAX=0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)

    #mlab.orientation_axes()
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig



def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def readIntoNumpy(fileName):
    tmp = []
    tmp2 = []
    with open(fileName) as f:
        for line in f :
            splitLine = line.rstrip().split()
            res = splitLine[8:15]   # [x y z h w l ry]
            # res[0:3], res[3:6] = res[3:6], res[0:3]
            # bug? it should be [h w l x y z ry]
            tmp.append(res)
            tmp2.append(splitLine[0])
    bboxes3d = np.array(tmp,  dtype=np.float32)
    labels = np.array(tmp2)
    return bboxes3d, labels

def pnt_in_cvex_hull_1(hull, pnt):
    #https://stackoverflow.com/questions/29311682/finding-if-point-is-in-3d-poly-in-python
    '''
    Checks if `pnt` is inside the convex hull.
    `hull` -- a QHull ConvexHull object
    `pnt` -- point array of shape (3,)
    '''
    new_hull = ConvexHull(np.concatenate((hull.points, [pnt])))
    if np.array_equal(new_hull.vertices, hull.vertices): 
        return True
    return False

if __name__ == "__main__":
    # path of lidar frame
    cloud_dir = "D:/data_object_velodyne/training/velodyne/"
    label_dir = "D:/data_object_label_2/training/label_2/"
    image_dir = "D:/"
    calib_dir = "D:/data_object_calib/training/calib"
    object_save_dir = "D:/data_object_box/"
    pcd_dir = "D:/data_object_pcd/"
    classes = ["Car", "Cyclist", "Pedestrian", "Van", "Truck", 
    "Person_sitting", "Tram", "Misc", "DontCare"] #full list
    """
    according to object3d.py's cls_type_to_id:
    it will be {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4},
    other classes will be -1
    """
    # classes = ["Car", "Pedestrian", "Cyclist"] #1,2,3

    for sample_id in tqdm(range(7481)):
        # path of output from model
        # bboxes3d_path = label_dir + "{:06d}.txt".format(sample_id)
        # bboxes3d, labels = readIntoNumpy(bboxes3d_path)
        # gt_obj_list = np.asarray(gt_obj_list)
    
        # cloud_path = cloud_dir + "{:06d}.bin".format(sample_id)
        # cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)
        # cloud = read_oxford_bin(cloud_path)
        # cloud = read_velodyne_bin(cloud_path)
        # corners3d = boxes3d_to_corners3d(bboxes3d)
        # boxes3d = np.zeros((len(gt_obj_list), 7))
        # for i in range(boxes3d.shape[0]):
        #     gt_obj = gt_obj_list[i]
        #     boxes3d[i] = [gt_obj.pos[0], gt_obj.pos[1], gt_obj.pos[2], 
        #                   gt_obj.w, gt_obj.l, gt_obj.h, gt_obj.ry]
        
        # point cloud
        calib = get_calib(calib_dir, sample_id)
        # img = self.get_image(sample_id)
        # img_shape = get_image_shape(image_dir, sample_id)
        pts_lidar = get_lidar(cloud_dir, sample_id)
        # get valid point (projected points should be in image)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
        # pts_intensity = pts_lidar[:, 3]
        np.savetxt(pcd_dir + "{:06d}.txt".format(sample_id), pts_rect)
        
        # label
        gt_obj_list = filtrate_objects(classes, get_label(label_dir, sample_id))
        boxes3d = objs_to_boxes3d(gt_obj_list)
        corners3d = boxes3d_to_corners3d(boxes3d)
        
        # print(sample_id)
        # print("cloud range", np.min(pts_lidar, axis=0), np.max(pts_lidar, axis=0))
        # print("calib cloud range", np.min(pts_rect, axis=0), np.max(pts_rect, axis=0))
        # print("box", boxes3d)
        # print("corner", corners3d)
        
        for obj_id, (gt_obj, corner3d) in enumerate(zip(gt_obj_list, corners3d)):
            label = gt_obj.cls_id
            
            if False:
                #https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
                # fail, only work for 2D
                polygon = Polygon(corner3d)
                in_hull = polygon.contains(pts_rect)
            elif False:
                #https://stackoverflow.com/questions/29311682/finding-if-point-is-in-3d-poly-in-python
                hull = ConvexHull(corner3d, incremental=True)
                in_hull = np.zeros(pts_rect.shape[0], dtype=bool)
                for i, pt in tqdm(enumerate(pts_rect)):
                    in_hull[i] = pnt_in_cvex_hull_1(hull, pt)
                # in_hull = np.asarray([pnt_in_cvex_hull_1(hull, pt) for pt in cloud], dtype=bool)
            else:
                #https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python
                corner2d = corner3d[:,[0,2]]
                corner2d = corner2d[:4]
                ymin = corner3d[:,1].min()
                ymax = corner3d[:,1].max()
                p = Path(corner2d) # make a polygon
                in_rect_2d = p.contains_points(pts_rect[:,[0,2]])
                in_hull = np.logical_and(in_rect_2d, pts_rect[:,1]>=ymin)
                in_hull = np.logical_and(in_hull, pts_rect[:,1]<=ymax)
            
            obj = pts_rect[in_hull]

            fname = object_save_dir + "{:06d}_{}_{}.txt".format(sample_id, obj_id, label)
            np.savetxt(fname, obj)
            
            # print("corner2d", corner2d)
            # print("y", ymin, ymax)
            # print("obj", obj.shape)
            # print(fname)

        # fig = draw_lidar(pts_rect)
        # fig = draw_gt_boxes3d(corners3d, fig)
        # mayavi.mlab.show()
    