# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SampleConsensusModels import SampleConsensusModel, SampleConsensusModelPlane
from RandomSampleConsensus import RandomSampleConsensus
from HoughTransform import HoughTransform
from LSQ import LSQ
import time
import warnings
import glob
import open3d as o3d

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
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
    return np.asarray(pc_list, dtype=np.float32)

def bin2pcd(path, is_write = False):
    #https://github.com/cuge1995/bin-to-pcd-python/blob/master/bin2pcd.py
    if path.endswith(".bin"):
        fnames = [path]
    else:
        fnames = glob.glob(path+'*.bin')
        
    for fname in fnames:
        np_pcd = read_velodyne_bin(fname)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pcd)
    
        if is_write:
            o3d.io.write_point_cloud(fname.replace(".bin", ".pcd"), pcd)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data, method = 0):
    # 作业1
    # 屏蔽开始
    m = data.shape[0]
    
    if method == 0:
        plane_model = SampleConsensusModelPlane(data)
        p = 0.99
        e = 1 - 47549/115384
        max_iterations = (int)(np.log(1-p)/np.log(1-pow(1-e, plane_model.points_need)))
        ransac = RandomSampleConsensus(plane_model, max_iterations=max_iterations)
        ransac.setDistanceThreshold(0.5)
        ransac.computeModel()
        inliers = ransac.getInliers()
    elif method == 1:
        ht = HoughTransform(model = 0, resolution = 0.4)
        ht.computeModel(data)
        inliers = ht.getInliers()
    elif method == 2:
        lsq = LSQ(model = 0, dist_thres = 0.1, loss_fnc = 2)
        lsq.computeModel(data)
        inliers = lsq.getInliers()
    
    masked = np.ma.array(np.arange(m), mask = False)
    masked.mask[inliers] = True
    outliers = masked.compressed()
    segmengted_cloud = data[outliers]
    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud, outliers

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def color_hex_to_rgb(hexes):
    rgbs = np.empty((hexes.shape[0], 3))
    
    for i, h in enumerate(hexes):
        h = h.lstrip('#')
        rgbs[i] = np.array(list(int(h[i:i+2], 16) for i in (0, 2, 4)))/255
   
    return rgbs
   
def plot_clusters_o3d(data, cluster_index, fname = None):
    #https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    rgbcolors = color_hex_to_rgb(colors[cluster_index])
    pcd.colors = o3d.utility.Vector3dVector(rgbcolors)
    
    if fname is None:
        o3d.visualization.draw_geometries([pcd])
    else:
        o3d.io.write_point_cloud(fname, pcd)

def main():
    root_dir = 'D:/PointCloudCourse/HW4/HomeworkIVclustering/clouds/' # 数据集路径
    # cat = os.listdir(root_dir)
    cat = glob.glob(root_dir+'*.bin')
    cat = cat[1:]
    # print(cat)
    iteration_num = len(cat)
    
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        # bin2pcd(filename, is_write = True)
        print('clustering pointcloud file:', filename)
        
        # open3d visualization
        pcd = o3d.io.read_point_cloud(filename.replace(".bin", ".pcd"))
        # o3d.visualization.draw_geometries([pcd])
        
        origin_points = read_velodyne_bin(filename)
        
        segmented_points, outliers = ground_segmentation(data=origin_points, method = 0)
        
        colors = np.zeros((origin_points.shape[0], 3))
        colors[outliers] = [1, 0, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
        
        X = segmented_points
        use_stdscaler = False
        
        if use_stdscaler:
            X = StandardScaler().fit_transform(X)    
        else:
            scale = np.max(np.max(X, axis=0) - np.min(X, axis=0))
            X -= np.mean(X, axis=0)
            X /= scale
        
        cluster_algo = 1
        
        if cluster_algo == 0:
            # estimate bandwidth for mean shift
            print("estimate bandwidth")
            bandwidth = cluster.estimate_bandwidth(X, quantile=0.3, n_samples = 1000, n_jobs=-1)
            bandwidth = 0.05
            print("bandwidth", bandwidth)
            print("MeanShift")
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
            algorithm = ms
            algorithm.fit(X)
            
            if hasattr(algorithm, 'labels_'):
                clusters_index = algorithm.labels_.astype(np.int)
            else:
                clusters_index = algorithm.predict(X)
            # print("y_pred", clusters_index)
            
            plot_clusters_o3d(segmented_points, clusters_index
                ,fname = filename[:-4] + "_ms_" + str(bandwidth) + ".pcd" 
                )
        elif cluster_algo == 1:
            for eps in [0.001, 0.003, 0.005]:
                for min_samples in [20]: #[3,5,10,20,30,50,60,70,80,90,100]:
                    print("eps", eps, "min_samples", min_samples)
                    dbscan = cluster.DBSCAN(eps=eps, 
                        min_samples = min_samples, n_jobs=-1)
                    dbscan.fit(X)
                    
                    algorithm = dbscan
                    
                    if hasattr(algorithm, 'labels_'):
                        clusters_index = algorithm.labels_.astype(np.int)
                    else:
                        clusters_index = algorithm.predict(X)
                    # print("y_pred", clusters_index)
            
                    plot_clusters_o3d(segmented_points, clusters_index
                        # ,fname = filename[:-4] + "_" + str(eps) + "_" + str(min_samples) + ".pcd" 
                        )
                    # plot_clusters(segmented_points, clusters_index)

if __name__ == '__main__':
    main()
