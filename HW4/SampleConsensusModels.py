# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:32:18 2020

@author: mimif
"""

import numpy as np

def normalized(a):
    # a : 1d or 2d numpy array
    if len(a.shape) == 1:
        return a / np.linalg.norm(a)
    elif len(a.shape) == 2:
        #https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
        l2 = np.atleast_1d(np.linalg.norm(a, axis=-1))
        l2[l2==0] = 1
        return a/np.expand_dims(l2, -1)
        

# mimic the API of PCL
class SampleConsensusModel(object):
    def __init__(self, data):
        self.data_ = data
    
    #https://stackoverflow.com/questions/4714136/how-to-implement-virtual-methods-in-python
    # serves as virtual method
    def computeModelCoefficients(self, samples):
        raise NotImplementedError()
        
    # serves as virtual method
    def getDistancesToModel(self):
        raise NotImplementedError()

class SampleConsensusModelPlane(SampleConsensusModel):
    def __init__(self, data):
        super().__init__(data)
        # need 3 points to determine a plane
        self.points_need = 3
    
    def computeModelCoefficients(self, samples):
        """
        a plane is defined by 
        3 points 
        or
        a point and a normal vector
        """
        self.p1 = samples[0]
        self.p2 = samples[1]
        self.p3 = samples[2]
        self.plane_normal = normalized(
            np.cross(self.p1-self.p2, self.p1-self.p3))
        return
    
    def getDistancesToModel(self):
        #https://mathworld.wolfram.com/Point-PlaneDistance.html
        distances = np.empty(self.data_.shape[0])
        for i, datum in enumerate(self.data_):
            distances[i] = np.linalg.norm(
                self.plane_normal.dot(datum-self.p1))
        return distances

class SampleConsensusModelLine(SampleConsensusModel):
    def __init__(self, data):
        super().__init__(data)
        # need 2 points to determine a line
        self.points_need = 2
    
    def computeModelCoefficients(self, samples):
        self.p1 = samples[0]
        self.p2 = samples[1]
        self.line_direction = normalized(samples[1]-samples[0])
        return
    
    def getDistancesToModel(self):
        #https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        distances = np.array(self.data_.shape[0])
        for i, datum in enumerate(self.data_):
            distances[i] = np.linalg.norm(
                np.cross(datum-self.p1, datum-self.p2))
        distances /= np.linalg.norm(self.p2-self.p1)
        return distances
    