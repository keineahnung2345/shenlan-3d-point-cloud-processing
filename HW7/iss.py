# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from glob import glob
import itertools
import os
from pyntcloud import PyntCloud
import open3d as o3d 

def get_neighbors(point_cloud_o3d, pcd_tree, radius):
    neighbors = []
    
    for point in point_cloud_o3d.points:
        cnt, idxs, dists = pcd_tree.search_radius_vector_3d(point, radius)
        neighbors.append(idxs)
    
    return neighbors

modelnet40_dir = "D:\modelnet40_normal_resampled"

classdirs = [os.path.join(modelnet40_dir, class_) \
             for class_ in os.listdir(modelnet40_dir) \
           if os.path.isdir(os.path.join(modelnet40_dir, class_))]

for classdir in classdirs[:3]:
    fnames = glob(classdir + "\*.txt")
    fname = fnames[np.random.randint(0, len(fnames))]
    point_cloud_pynt = PyntCloud(
        pd.DataFrame(np.loadtxt(fname, delimiter=','),
                     columns = ['x','y','z', 'red', 'green', 'blue']))
    red = np.array([255, 0, 0], dtype=np.uint8)
    black = np.array([0, 0, 0], dtype=np.uint8)
    #https://github.com/daavoo/pyntcloud/issues/155
    point_cloud_pynt.points.loc[:, ['red', 'green', 'blue']] = black
    
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    
    points = point_cloud_pynt.points
    points = points.to_numpy()
    print('total points number is:', points.shape[0])
    
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    
    # the range of the cloud in three dimensions
    cloud_range = points.max(axis=0)-points.min(axis=0)
    
    salient_radius_multiplier_range = [0.05] 
    #np.concatenate([
    #    np.arange(0.03, 0.1, 0.02), 
    #    np.arange(0.1, 1.0, 0.2)])
    non_max_radius_multiplier_range = [0.09] #salient_radius_multiplier_range
    gamma_21_range = [0.975]#np.arange(0.3, 1.0, 0.2)
    gamma_32_range = [0.975]#np.arange(0.3, 1.0, 0.2)
    
    for srm, nmrm, gamma_21, gamma_32 in itertools.product(salient_radius_multiplier_range, 
                               non_max_radius_multiplier_range, 
                               gamma_21_range,
                               gamma_32_range
                               ):
        srm = round(srm, 2)
        nmrm = round(nmrm, 2)
        gamma_21 = round(gamma_21, 3)
        gamma_32 = round(gamma_32, 3)
        salient_radius = cloud_range.max() * srm
        non_max_radius = cloud_range.max() * nmrm
        
        """
        calculate covariance matrix
        """
        neighbors_salient_radius = get_neighbors(point_cloud_o3d, pcd_tree, salient_radius)
        
        # key: index of the point
        # value: (score(lambda3), keep(used by NMS))
        candidates = {}
        
        for i in range(len(point_cloud_o3d.points)):
            numerator = np.zeros((3,3))
            wsum = 0
            for j in neighbors_salient_radius[i]:
                wj = 1/len(neighbors_salient_radius[j])
                diff = point_cloud_o3d.points[j] - point_cloud_o3d.points[i]
                numerator += wj * np.outer(diff, diff)
                wsum += wj
            cov = numerator / wsum
            
            # need to substract mean before SVD?
            u, s, vh = np.linalg.svd(cov, full_matrices=True)
            eigenvalues = s
            eigenvectors = u
            
            # decreasing order
            eigenvalues = sorted(eigenvalues, reverse=True)
            
            lambda1, lambda2, lambda3 = eigenvalues
            
            keep = lambda2/lambda1 < gamma_21 and lambda3/lambda2 < gamma_32
            
            candidates[i] = [lambda3, keep]
            
        
        """
        NMS
        """
        #https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
        candidates = dict(sorted(candidates.items(), 
                                 key=lambda item: item[1][0],
                                 reverse = True))
        
        
        neighbors_non_max = get_neighbors(point_cloud_o3d, pcd_tree, non_max_radius)
        
        for idxi, (scorei, keepi) in candidates.items():
            # this candidate is already ruled out
            if not keepi: continue
        
            # suppress neighbors in radius r
            for idxj in neighbors_non_max[idxi]:
                if idxj == idxi: continue
                candidates[idxj][1] = False
        
        feature_point_idxs = []
        for idxi, (scorei, keepi) in candidates.items():
            if keepi:
                point_cloud_pynt.points.loc[idxi, ['red', 'green', 'blue']] = red
                feature_point_idxs.append(idxi)
        
        """
        log
        """
        fname = fname.replace('\\', '/').rsplit('/')[-1]
        fname = fname[:fname.find('.')]
        
        point_cloud_pynt.points.loc[feature_point_idxs,:].to_csv(
            "output/{}_{}_{}_{}_{}.txt".format(fname, srm, nmrm, gamma_21, gamma_32), 
            header=False, index=False)
        
        # with open("output/stats.txt", "a") as f:
        #     l = [class_, srm, nmrm, gamma_21, gamma_32, len(feature_point_idxs)]
        #     l = [str(e) for e in l]
        #     f.write(','.join(l)+'\n')
        print(fname, nmrm, gamma_21, gamma_32, 
              "There are", len(feature_point_idxs), "feature points")

#%%
"""
Visualization
"""

point_cloud_pynt.plot()
# scene = point_cloud_pynt.plot(initial_point_size=0.01, return_scene=True)
# point_cloud_pynt_feature_points = PyntCloud(point_cloud_pynt.points.iloc[feature_point_idxs,:])
# scene.children += [point_cloud_pynt_feature_points]
# point_cloud_pynt_feature_points.plot(initial_point_size=0.5, scene = scene)
point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
o3d.visualization.draw_geometries([point_cloud_o3d])
