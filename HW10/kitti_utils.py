# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:54:54 2021

@author: mimif
"""

import os
import numpy as np
import object3d
import calibration
from PIL import Image

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [object3d.Object3d(line) for line in lines]
    # for i in range(len(objects)):
    #     objects[i].pos[1], objects[i].pos[2] = objects[i].pos[2], objects[i].pos[1]
    return objects

def get_lidar(lidar_dir, idx):
    lidar_file = os.path.join(lidar_dir, '%06d.bin' % idx)
    assert os.path.exists(lidar_file)
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

def get_label(label_dir, idx):
    if idx < 10000:
        label_file = os.path.join(label_dir, '%06d.txt' % idx)
    # else:
    #     label_file = os.path.join(aug_label_dir, '%06d.txt' % idx)

    assert os.path.exists(label_file)
    # return kitti_utils.get_objects_from_label(label_file)
    return get_objects_from_label(label_file)

def get_calib(calib_dir, idx):
    calib_file = os.path.join(calib_dir, '%06d.txt' % idx)
    assert os.path.exists(calib_file)
    return calibration.Calibration(calib_file)


def get_image_shape(image_dir, idx):
    img_file = os.path.join(image_dir, '%06d.png' % idx)
    assert os.path.exists(img_file)
    im = Image.open(img_file)
    width, height = im.size
    return height, width, 3
    
def filtrate_objects(classes, obj_list):
    """
    Discard objects which are not in self.classes (or its similar classes)
    :param obj_list: list
    :return: list
    """
    type_whitelist = classes

    valid_obj_list = []
    for obj in obj_list:
        if obj.cls_type not in type_whitelist:  # rm Van, 20180928
            continue
        valid_obj_list.append(obj)
    return valid_obj_list

def objs_to_boxes3d(obj_list):
    boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
    for k, obj in enumerate(obj_list):
        boxes3d[k, 0:3], boxes3d[k, 3], boxes3d[k, 4], boxes3d[k, 5], boxes3d[k, 6] \
            = obj.pos, obj.h, obj.w, obj.l, obj.ry
    return boxes3d

def boxes3d_to_corners3d(boxes3d, rotate=True):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :param rotate:
    :return: corners3d: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    h, w, l = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dtype=np.float32).T  # (N, 8)
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T  # (N, 8)

    y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
    y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)

    if rotate:
        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
        rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                              [zeros,       ones,       zeros],
                              [np.sin(ry), zeros,  np.cos(ry)]])  # (3, 3, N)
        # rot_list = np.array([[np.cos(ry), zeros, np.sin(ry)],
        #                      [zeros,       ones,       zeros],
        #                      [-np.sin(ry), zeros,  np.cos(ry)]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                       z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)

def save_kitti_format(classes, sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape):
    corners3d = boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (classes[k], alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]), file=f)