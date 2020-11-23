# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:17:40 2020

@author: mimif
"""

# 文件功能： 实现 K-Means 算法

import numpy as np
from tqdm import tqdm
import time
from sklearn.neighbors import KDTree
import numpy.linalg as LA
from sklearn.cluster import KMeans

class SpectralClustering(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, nnk = 3, nnradius = 1, 
                 normalized = True, use_radius_nn = False, 
                 use_gauss_dist = False, gauss_sigma = 5e-1):
        self.k_ = n_clusters
        # 屏蔽开始
        # the k for KNN
        self.nnk_ = nnk
        self.nnradius_ = nnradius
        self.labels_ = np.empty(0)
        self.normalized_ = normalized
        self.use_radius_nn_ = use_radius_nn
        self.use_gauss_dist_ = use_gauss_dist
        self.gauss_sigma_ = gauss_sigma
        # 屏蔽结束

    def gauss_(self, x):
        return np.exp(-x*x/(2*self.gauss_sigma_*self.gauss_sigma_))
        
    def fit(self, data):
        # 屏蔽开始
        
        # data: m * dim array
        m = data.shape[0]
        # print("m", m)
        
        tree = KDTree(data)
        
        W = np.zeros((m, m))
        
        for di, datum in enumerate(data):
            # neighbors' index
            if self.use_radius_nn_:
                nis, ndists = tree.query_radius([datum], self.nnradius_, 
                                                return_distance=True)
            else:
                # the order of return value is different from query_radius!
                ndists, nis = tree.query([datum], self.nnk_+1, 
                                         return_distance=True)
            
            nis = nis[0]
            ndists = ndists[0]
            # print("indices", nis)
            # print("ndists", ndists)
            # print(nis.shape)
            
            # if len(nis.shape) == 0: continue
            # print(di, nis, ndists)
            # print("neighbors",nis.shape)
            for ni, ndist in zip(nis, ndists):
                # the point itself will be one of its knn, need to skip it
                if ni == di: continue
                if self.use_gauss_dist_:
                    W[di][ni] = W[ni][di] = self.gauss_(ndist)
                else:
                    W[di][ni] = W[ni][di] = 1/ndist
        
        D = np.diag(W.sum(axis=1))
        
        # unnormalized Laplacian
        L = D - W
        
        # for debugging
        self.W = W
        self.D = D
        
        if self.normalized_:
            L = a = np.matmul(LA.inv(D), L)
            L = b = np.identity(m) - np.matmul(LA.inv(D), W)
            assert(np.allclose(a,b))
            
        # for debugging
        self.L = L
        
        eigvals, eigvecs = LA.eig(L)
        """
        From numpy.linalg.eig's doc:
        The eigenvalues are not necessarily ordered!!
        so we need to sort eigen values!!
        """
        sorted_idx = np.argsort(eigvals)
        # smallest self.k_ eigenvectors
        V = eigvecs[:, sorted_idx[:self.k_]]
        
        # for debugging
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.V = V
        
        # run kmeans
        self.labels_ = KMeans(n_clusters=self.k_).fit_predict(V)
        # 屏蔽结束

    def predict(self, p_datas):
        pass
    
if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    sc = SpectralClustering(n_clusters=2)
    sc.fit(x)

    cat = sc.labels_
    print(cat)

