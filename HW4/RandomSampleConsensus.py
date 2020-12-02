# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:25:24 2020

@author: mimif
"""

import numpy as np
from tqdm import tqdm
from SampleConsensusModels import SampleConsensusModel

class RandomSampleConsensus(object):
    def __init__(self, model : SampleConsensusModel, 
                 dist_thres : float = 0.01, max_iterations : int = 1000):
        self.model_ = model
        self.dist_thres_ = dist_thres
        self.max_iterations_ = max_iterations
    
    def setDistanceThreshold(self, dist_thres):
        self.dist_thres_ = dist_thres
    
    def computeModel(self):
        max_inlier_count = 0
        model = self.model_
        for _ in tqdm(range(self.max_iterations_)):
            sample = model.data_[np.random.choice(model.data_.shape[0], model.points_need, replace = False), :]
            model.computeModelCoefficients(sample)
            distances = model.getDistancesToModel()
            inlier_count = np.count_nonzero(distances <= self.dist_thres_)
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                self.model_ = model
                self.inliers_ = np.where(distances <= self.dist_thres_)
    
    def getInliers(self):
        return self.inliers_


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    ransac = RandomSampleConsensus()
    ransac.fit(x)

    inliers = ransac.predict()
    print(inliers)


    