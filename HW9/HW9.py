# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:24:24 2020

@author: mimif
"""

import struct
import numpy as np
import glob
import os
import numpy as np
import numpy.linalg as LA
import pandas as pd
import open3d as o3d
from functools import partial
from pyntcloud import PyntCloud
import registration_dataset.evaluate_rt
from iss import iss
from HW8 import fpfh_calculator

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

def get_neighbors(cloud, radius):
    tree = o3d.geometry.KDTreeFlann(cloud)
    neighbors = []
    
    for pi, point in enumerate(cloud.points):
        cnt, idxs, dists = tree.search_radius_vector_3d(point, radius)
        # here we find a point from cloud A in cloud B
        # idxs.remove(pi)
        neighbors.append(idxs)
    
    return neighbors

def lrf_builder(point_cloud_o3d, neighbors_indices, R, center):
    """
    build LRF
    """
    denom = 0
    M = np.zeros((3, 3))
    for i2 in neighbors_indices:
        p2 = point_cloud_o3d.points[i2]
        denom += (R - LA.norm(center-p2))
        M += (R - LA.norm(center-p2)) * np.outer(p2-center, (p2-center).T)
    assert denom != 0, "neighbor cnt: {}, denom: {}".format(
        len(neighbors_indices), denom)
    M /= denom
    u, s, vh = np.linalg.svd(M, full_matrices=True)
    eigenvalues = s
    eigenvectors = u
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    x_pos, y_pos, z_pos = eigenvectors
    Sx_pos, Sz_pos = 0, 0
    
    for i2 in neighbors_indices:
        p2 = point_cloud_o3d.points[i2]
        if np.dot(p2-center, x_pos) >= 0:
            Sx_pos += 1
        if np.dot(p2-center, z_pos) >= 0:
            Sz_pos += 1
    
    # if there are more points whose x >= 0, then x will be x_pos
    x = x_pos * pow(-1, Sx_pos < len(neighbors_indices)/2)
    z = z_pos * pow(-1, Sz_pos < len(neighbors_indices)/2)
    y = np.cross(z, x)
    
    # normalize the  axis!
    x = x / LA.norm(x)
    y = y / LA.norm(y)
    z = z / LA.norm(z)
    
    return np.array([x, y, z])

def transformMatrixFrom2Vectors(src_vec, tgt_vec):
    #https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(src_vec, tgt_vec)
    print("v", v)
    s = LA.norm(v)
    c = np.dot(src_vec, tgt_vec)
    v_mat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
        ])
    print("v_mat", v_mat)
    return np.identity(3) + v_mat + np.dot(v_mat, v_mat) *(1-c)/(s*s)

tgt_tree = None
radius = None
association = {}
icp_dist_thres = float("inf") #100
log = np.zeros((500, 4))

def ICP(src_cloud, tgt_cloud, save_path = None):
    global tgt_tree
    global log
    R, t = np.identity(3), np.zeros((3,1))
    #homogeneous transform matrix, 
    #recoding the accumulated rotation and translation
    homo = np.vstack((np.hstack((R,t)),[0,0,0,1]))
    R_delta, t_delta = float("inf"), float("inf")
    R_thres, t_thres = 1e-5, 1e-4
    cost = float("inf")
    
    src_tree = o3d.geometry.KDTreeFlann(src_cloud)
    tgt_tree = o3d.geometry.KDTreeFlann(tgt_cloud)
    src_idxs = range(len(src_cloud.points))
    
    # print("src", len(src_cloud.points))
    # print("tgt", len(tgt_cloud.points))
    
    """
    initilization
    use PCA to fin
    """
    initialize_use_lrf = True
    
    if initialize_use_lrf:
        src_neighbors_indices = range(len(src_cloud.points))
        src_R = np.max(np.max(src_cloud.points)-np.min(src_cloud.points))/2
        src_center = (np.max(src_cloud.points, axis=0)+np.min(src_cloud.points, axis=0))/2
        src_coord = lrf_builder(src_cloud, src_neighbors_indices, src_R, src_center)
        
        tgt_neighbors_indices = range(len(tgt_cloud.points))
        tgt_R = np.max(np.max(tgt_cloud.points)-np.min(tgt_cloud.points))/2
        tgt_center = (np.max(tgt_cloud.points, axis=0)+np.min(tgt_cloud.points, axis=0))/2
        tgt_coord = lrf_builder(tgt_cloud, tgt_neighbors_indices, tgt_R, tgt_center)
    
        # R = transformMatrixFrom2Vectors(src_coord, tgt_coord)
        #https://stackoverflow.com/questions/55082928/change-of-basis-in-numpy
        R = np.matmul(tgt_coord, LA.inv(src_coord))
        t = (tgt_center - src_center)[...,np.newaxis]
        # print("R", R.shape)
        # print("t", t.shape)
        homo = np.vstack((np.hstack((R,t)),[0,0,0,1]))
        src_points = np.asarray(src_cloud.points).T
        # print("src_points", src_points.shape)
        # print("matmul", np.matmul(R, src_points).shape)
        src_points = np.asarray(np.matmul(R, src_points) + t)
        point_cloud_pynt = PyntCloud(
            pd.DataFrame(src_points.T, 
                         columns = ['x','y','z']))
        src_normals = src_cloud.normals
        src_cloud = point_cloud_pynt.to_instance("open3d", mesh=False)
        src_cloud.normals = src_normals
        
        if save_path:
            np.savetxt(save_path(i="pca"),
                       src_points.T, delimiter=',')
    
    use_fea_detect = False
    if use_fea_detect:
        srm = 0.05
        nmrm = 0.09 #salient_radius_multiplier_range
        gamma_21 = 0.975
        gamma_32 = 0.975
        
        src_idxs = iss(src_cloud, src_tree, 
                                 srm, nmrm, gamma_21, gamma_32)
        
        tgt_idxs = iss(tgt_cloud, tgt_tree, 
                                 srm, nmrm, gamma_21, gamma_32)
        # only use feature point for data association
        tgt_pt_np = np.take(np.asarray(tgt_cloud.points),
                    tgt_idxs, axis=0)
        tgt_nm_np = np.take(np.asarray(tgt_cloud.normals),
                    tgt_idxs, axis=0)
        point_cloud_pynt = PyntCloud(
            pd.DataFrame(tgt_pt_np, 
                         columns = ['x','y','z']))
        tgt_cloud = point_cloud_pynt.to_instance("open3d", mesh=False)
        tgt_cloud.normals = o3d.cpu.pybind.utility.Vector3dVector(
            tgt_nm_np)
        tgt_tree = o3d.geometry.KDTreeFlann(tgt_cloud)
    
    use_fea_descript = False
    fea_descript_thres = 0
    if use_fea_descript:
        r = 0.03
        src_fea = fpfh_calculator(src_cloud, radius = r)
        tgt_fea = fpfh_calculator(tgt_cloud, radius = r)
        
    src_points = np.asarray(src_cloud.points).T
    tgt_points = np.asarray(tgt_cloud.points).T
    # print(src_points.shape)
    
    last_dist_avg = 0
    farthest_point = (-1, 0) #(src_i, dist)
    iteration = 0
    
    while iteration < 500 and \
        (R_delta > R_thres or t_delta > t_thres): # and cost >= 0.01
        """
        Data association
        """
        global association
        association = {}
        src_center, tgt_center = 0.0, 0.0
        # use tgt_tree
        dist_avg = 0
        
        farthest_point = (-1, 0) #(src_i, dist)
        # for src_i in range(src_points.shape[1]): #3*N
        for src_i in src_idxs:
            src_pt = src_points[:,src_i]
            cnt, idxs, dists = tgt_tree.search_knn_vector_3d(src_pt, 1)
            tgt_i = idxs[0]
            # remove outlier pairs
            dist_avg += dists[0]
            if dists[0] > farthest_point[1]:
                farthest_point = src_i, dists[0]
            if dists[0] > icp_dist_thres:
                continue
            if use_fea_descript:
                if np.dot(src_fea[src_i], tgt_fea[tgt_i]) < fea_descript_thres: 
                    continue
            association[src_i] = tgt_i
            src_center += src_pt
            tgt_center += tgt_points[:,tgt_i]
        # print("association: ", len(association))
        # print("association keys: ", len(association.keys()))
        # print("farthest point: ", farthest_point)
        # print("association values: ", len(association.values()))
        dist_avg = dist_avg/len(association.keys())
        # print("average dist:", dist_avg)
        last_dist_avg = dist_avg
        """
        solve R and t
        """
        pt2pl = True
        
        if not pt2pl:
            # point to point
            src_center = (src_center/len(association))[...,np.newaxis]
            tgt_center = (tgt_center/len(association))[...,np.newaxis]
            # transpose to 3*N
            src_norm_cloud = np.take(src_points,
                                     list(association.keys()), axis=1)#.T
            # transpose to 3*N
            tgt_norm_cloud = np.take(tgt_points,
                                     list(association.values()), axis=1)#.T
            src_norm_cloud = src_norm_cloud - src_center #3*N
            tgt_norm_cloud = tgt_norm_cloud - tgt_center #3*N
            
            # print("matrix to svd:", np.matmul(tgt_norm_cloud, src_norm_cloud.T).shape)
            # to large matrix leads to "init_dgesdd failed init"
            u, s, vh = np.linalg.svd(
                np.matmul(tgt_norm_cloud, src_norm_cloud.T),
                full_matrices=True)
            
            R_delta = LA.norm(np.matmul(u, vh) - R)
            R = np.matmul(u, vh)
            t_delta = LA.norm(tgt_center - np.matmul(R, src_center) - t)
            t = tgt_center - np.matmul(R, src_center)
        else:
            # point to plane
            tgt_normals = np.asarray(tgt_cloud.normals)
            # print(tgt_normals.shape)
            # print(src_points.shape)
            ax1 = (tgt_normals[:,2]*src_points[1,:] - tgt_normals[:,1]*src_points[2,:])[...,np.newaxis]
            ax2 = (tgt_normals[:,0]*src_points[2,:] - tgt_normals[:,2]*src_points[0,:])[...,np.newaxis]
            ax3 = (tgt_normals[:,1]*src_points[0,:] - tgt_normals[:,0]*src_points[1,:])[...,np.newaxis]
            # print(ax1.shape)
            A = np.hstack([ax1, ax2, ax3, tgt_normals])
            #https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
            b = np.einsum('ij,ij->i', tgt_normals, (tgt_points - src_points).T)
            # b = np.inner1d(src_normals, (tgt_points - src_points).T)
            x = np.matmul(
                    np.matmul(
                        LA.inv(np.matmul(A.T,A)), 
                        A.T), 
                    b)
            alpha, beta, gamma, tx, ty, tz = x
            R_tmp = np.array([
                [1, -gamma, beta],
                [gamma, 1, -alpha],
                [-beta, alpha, 1]])
            R_delta = LA.norm(R_tmp - R)
            R = R_tmp
            
            t_tmp = np.array([tx, ty, tz])[...,np.newaxis]
            t_delta = LA.norm(t_tmp - t)
            t = t_tmp
        
        # print("R", R)
        # print("t", t)
        # print("R delta", R_delta)
        # print("t delta", t_delta)
        
        # print("R:",R.shape)
        # print("t:",t.shape)
        src_points = np.matmul(R, src_points) + t
        
        src_matched_cloud = src_points[:,list(association.keys())]
        tgt_matched_cloud = tgt_points[:,list(association.values())]
        cost = LA.norm(src_matched_cloud - tgt_matched_cloud)/len(association)
        print("cost", cost)
        
        log[iteration] = [R_delta, t_delta, cost, dist_avg]
        
        # print("homo", homo)
        # print("cur homo", np.vstack((np.hstack((R,t)),[0,0,0,1])))
        homo = np.matmul(
            np.vstack((np.hstack((R,t)),[0,0,0,1])),
            homo)
        # print("homo", homo)
        
        if save_path:
            if iteration % 10 == 0:
                np.savetxt(save_path(i=str(iteration)+"_"+str(round(cost, 5))),
                           src_points.T, delimiter=',')
            np.savetxt(save_path(i="final"),
                       src_points.T, delimiter=',')
        iteration += 1
    
    np.savetxt(save_path(i="log"),
               log, delimiter=',',
               header="R_delta,t_delta,cost,dist_avg",
               fmt='%f,%f,%f,%f')
    
    return src_points, homo

def copysign(v, s):
    # copy the sign of s to v
    if v * s < 0:
        v *= -1
    return v
    
def rotmat2quaternion(m):
    #https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    trace = m[0][0]+m[1][1]+m[2][2]
    qw = np.sqrt(max(0, trace+1))/2
    qx = np.sqrt(max(0, 1+m[0][0]-m[1][1]-m[2][2]))/2
    qy = np.sqrt(max(0, 1-m[0][0]+m[1][1]-m[2][2]))/2
    qz = np.sqrt(max(0, 1-m[0][0]-m[1][1]+m[2][2]))/2
    qx = copysign(qx, m[2][1]-m[1][2])
    qy = copysign(qy, m[0][2]-m[2][0])
    qz = copysign(qz, m[1][0]-m[0][1])
    return qw, qx, qy, qz

reg_result = np.genfromtxt("registration_dataset/reg_result.txt", delimiter=',')
reg_result = reg_result[1:]

pairs = [(int(idx1), int(idx2)) for idx1, idx2 in reg_result[:,:2]]

fname = "D:\\PointCloudCourse\\HW9\\registration_dataset\\point_clouds\\{}.bin"
resfname = "D:\\PointCloudCourse\\HW9\\result.csv"
icpfname = "D:\\PointCloudCourse\\HW9\\ICP\\{s}_{t}_{i}.txt".format
result = np.zeros((len(pairs), 9))

# for i, (src_idx, tgt_idx) in enumerate(pairs):
# #from evaluate_rt.py, it seems we need to transform dst to src?
for i, (tgt_idx, src_idx) in enumerate(pairs):
    print("=============", i, ":", src_idx, tgt_idx, "=============")
    # src_idx, tgt_idx = tgt_idx, src_idx
    src_fname = fname.format(src_idx)
    tgt_fname = fname.format(tgt_idx)
    src_np = read_oxford_bin(src_fname).T
    tgt_np = read_oxford_bin(tgt_fname).T
    
    point_cloud_pynt = PyntCloud(
        pd.DataFrame(src_np[:,:3], 
                     columns = ['x','y','z']))
    src_cloud = point_cloud_pynt.to_instance("open3d", mesh=False)
    src_cloud.normals = o3d.cpu.pybind.utility.Vector3dVector(src_np[:,3:])
    
    point_cloud_pynt = PyntCloud(
        pd.DataFrame(tgt_np[:,:3], 
                     columns = ['x','y','z']))
    tgt_cloud = point_cloud_pynt.to_instance("open3d", mesh=False)
    tgt_cloud.normals = o3d.cpu.pybind.utility.Vector3dVector(tgt_np[:,3:])
    
    # tgt_tree = o3d.geometry.KDTreeFlann(tgt_cloud)
    
    radius = 0.03
    
    transformed_src_cloud, homo_mat = ICP(
        src_cloud, #np.asarray(src_cloud.points).T, 
        tgt_cloud, #np.asarray(tgt_cloud.points).T,
        partial(icpfname, s=src_idx, t=tgt_idx))
    
    tx, ty, tz = homo_mat[:3,3]
    rot_mat = homo_mat[:3,:3]
    qw, qx, qy, qz = rotmat2quaternion(rot_mat)
    
    # result[i] = [src_idx, tgt_idx, tx, ty, tz, qw, qx, qy, qz]
    result[i] = [tgt_idx, src_idx, tx, ty, tz, qw, qx, qy, qz]

np.savetxt(resfname, result, delimiter=',', 
            header="idx1,idx2,t_x,t_y,t_z,q_w,q_x,q_y,q_z",
            fmt='%i,%i,%f,%f,%f,%f,%f,%f,%f')

