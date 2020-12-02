# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:59:18 2020

@author: mimif
"""

import numpy as np
from tqdm import tqdm
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

class LSQ(object):
    def __init__(self, model = 0, 
                 dist_thres : float = 0.1,
                 loss_fnc : int = 0):
        self.model_ = model
        self.param_ = None
        self.dist_thres_ = dist_thres
        self.inliers_ = []
        self.huber_delta_ = 1.0
        self.loss_fnc_ = loss_fnc

    def cauchy(self, x, A, b):
        x = np.reshape(x, (3,1))
        # print("x", x.shape)
        # print("A", A.shape)
        # print("b", b.shape)
        # return np.sum(np.log(1+np.abs(A*x - b)))
        err = A*x - b
        res = np.log(1+np.abs(err))
        res = np.array(res).flatten()
        # print("res", res.shape)
        return res
    
    def huber(self, x, A, b):
        x = np.reshape(x, (3, 1))
        res = np.zeros(A.shape[0])
        err = A*x - b
        err = np.array(err).flatten()
        err_lt_idxs = np.where(np.abs(err) < self.huber_delta_)[0]
        err_ge_idxs = np.where(np.abs(err) >= self.huber_delta_)[0]
        res[err_lt_idxs] = np.power(err[err_lt_idxs], 2)
        res[err_ge_idxs] = 2 * self.huber_delta_* \
            (np.abs(err[err_ge_idxs]) - self.huber_delta_/2)
        return res
    
    def computeModelPlane_(self, data):
        """
        z = ax+by+c
        """
        m = data.shape[0]
        
        X = data
        X = StandardScaler().fit_transform(X)
        
        # data's first 2 columns + a column of 1s
        A = np.c_[X[:, :2], np.ones(m)]
        b = X[:, 2:]
        
        A = np.matrix(A)
        b = np.matrix(b)
        
        print("A", A.shape)
        print("b", b.shape)
        
        self.param_ = np.linalg.inv(A.T * A) * A.T * b
        # self.param_ = np.reshape(self.param_, (3, 1))
        print("L2: ", self.param_.shape, self.param_)
        
        if self.loss_fnc_ == 1:
            self.param_, cov, infodict, msg, ier = leastsq(self.cauchy, 
                    self.param_, args=(A, b), full_output=True)
            print("cauchy: ", self.param_)
        elif self.loss_fnc_ == 2:
            self.param_, cov, infodict, msg, ier = leastsq(self.huber, 
                    self.param_, args=(A, b), full_output=True)
            print("huber: ", self.param_)
        
        self.param_ = np.array(self.param_)
        self.inliers_ = []
        dists = []
        for i, datum in enumerate(X):
            dist = np.linalg.norm(datum.dot(self.param_))
            dists.append(dist)
            if dist < self.dist_thres_:
                self.inliers_.append(i)
        
        plt.hist(dists)
    
    def computeModel(self, data):
        if self.model_ == 0:
            self.computeModelPlane_(data)
    
    def getInliers(self):
        return self.inliers_

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    # ransac = RandomSampleConsensus()
    # ransac.fit(x)

    # inliers = ransac.predict()
    # print(inliers)

