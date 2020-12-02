# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:32:43 2020

@author: mimif
"""

import numpy as np
from tqdm import tqdm
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
import matplotlib.pyplot as plt

class HoughTransform(object):
    def __init__(self, model : 0, resolution : float):
        self.model_ = model
        self.resolution_ = resolution
        self.inliers_ = []
    
    def computeModelPlane_(self, data):
        """
        https://robotik.informatik.uni-wuerzburg.de/telematics/download/3dresearch2011.pdf
        plane : z = mx*x+my*y+rho
        -> rho = z-mx*x-my*y
        """
        
        X = data
        X = StandardScaler().fit_transform(X)
        # X = MinMaxScaler().fit_transform(X)
        # X -= np.mean(X, axis=0)
        # X_range = (X.max(axis=0) - X.min(axis=0)).max()
        # X /= X_range
        
        # 3d dict counter
        grid2count = defaultdict(lambda : 
                        defaultdict(lambda : 
                            defaultdict(lambda : [])))
        
        for mx in np.arange(-1.0, 1.0, self.resolution_):
            for my in tqdm(np.arange(-1.0, 1.0, self.resolution_)):
                for i, (x,y,z) in enumerate(X):
                    rho = z-mx*x-my*y
                    # take floor
                    rho = rho//self.resolution_*self.resolution_
                    # inliers
                    grid2count[mx][my][rho].append(i)
        
        # names = []
        # cnts = []
        for mx, xgrids in grid2count.items():
            for my, ygrids in xgrids.items():
                for rho, inliers in ygrids.items():
                    # names.append(str(round(mx,2))+"_"+
                    #              str(round(my,2))+"_"+str(round(rho,2)))
                    # cnts.append(len(inliers))
                    # print(mx, my, rho, cnts[-1])
                    if len(inliers) > len(self.inliers_):
                        self.inliers_ = inliers
                        self.param_ = (mx, my, rho)
        
        # print(sum(cnts), "/", X.shape[0])
        # plt.rcParams.update({'font.size': 5})
        # plt.barh(names, cnts)
    
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


    