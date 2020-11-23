# 文件功能： 实现 K-Means 算法

import numpy as np
from tqdm import tqdm
import time

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300, fit_method = 3):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        # 屏蔽开始
        # suppose every datum is a length-dim vector
        # then self.centers_ is a self.k_ * dim 2d array
        self.centers_ = np.empty((self.k_, 0))
        self.fit_method_ = fit_method
        # 屏蔽结束

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        
        # data: m * dim array
        m = data.shape[0]
        dim = data.shape[1]
        
        # random pick k points as the center of clusters
        # np.random.choice(m, self.k_, replace=False)： random choose indices
        self.centers_ = data[np.random.choice(m, self.k_, replace=False)]
        
        fit_begin = time.time()
        estep_total = 0
        mstep_total = 0
        
        """
        method 0~3: corresponds to different degree of optimization
        method 0: use two for loops, slow ~ 20s
        method 1: calculate the distances of datum with all centers at one time ~15s
        method 2: calculate the distances of a center with all data at one time ~6s
        method 3: optimize from method 2, use smaller loop ~0.24s
        """
        
        last_distortion = float("inf")
        for cur_iter in range(self.max_iter_):
            sums = np.zeros((self.k_, dim))
            counts = np.zeros(self.k_, dtype=int)
            distortion = 0
            assigned = np.empty(m, dtype=int)
            
            # E-step: assign points to k clusters
            # used to store indices
            estep_begin = time.time()
            
            if self.fit_method_ == 0:
                # use two for loops, slowest
                for di, datum in enumerate(data):
                    mindist = float("inf")
                    for ci, center in enumerate(self.centers_):
                        dist = np.linalg.norm(datum - center)
                        if dist < mindist:
                            mindist = dist
                            assigned[di] = ci
                    sums[assigned[di]] += datum
                    counts[assigned[di]] += 1
                    distortion += mindist*mindist
            elif self.fit_method_ == 1:
                # calculate the distances of datum with all centers at one time
                for di, datum in enumerate(data):
                    dists = np.linalg.norm(self.centers_-datum, axis=1)
                    assigned[di] = np.argmin(dists)
                    mindist = dists[assigned[di]]
                    
                    sums[assigned[di]] += datum
                    counts[assigned[di]] += 1
                    distortion += mindist*mindist
            elif self.fit_method_ == 2:
                # because self.k_ << m, we can speed up by converting from
                # "calculating distances of a datum with all centers" * m time to
                # "calculating distances of a center with all data" * self.k_ times
                
                # record the distances of every datum to every centers
                dist_mat = np.empty((m, self.k_))
                for ci, center in enumerate(self.centers_):
                    dist_mat[:,ci] = np.linalg.norm(data-center, axis=1)
                
                for di, datum in enumerate(data):
                    assigned[di] = np.argmin(dist_mat[di])
                    sums[assigned[di]] += datum
                    counts[assigned[di]] += 1
                    distortion += dist_mat[di][assigned[di]] * dist_mat[di][assigned[di]]
            elif self.fit_method_ == 3:
                dist_mat = np.empty((m, self.k_))
                for ci, center in enumerate(self.centers_):
                    dist_mat[:,ci] = np.linalg.norm(data-center, axis=1)
                
                # optimize from method 2, not using O(m) loop
                assigned = np.argmin(dist_mat, axis=1)
                distortion = (np.min(dist_mat, axis=1)**2).sum()
                
                # O(self.k_) loop << O(m) loop
                for ci in range(self.k_):
                    counts[ci] = np.count_nonzero(assigned==ci)
                    sums[ci] = data[np.where(assigned==ci)].sum(axis=0)
                
            estep_total += time.time() - estep_begin
                
            # M-step: recalculate the center of k clusters
            mstep_begin = time.time()
            for i in range(self.k_):
                self.centers_[i] = sums[i]/counts[i]
            mstep_total += time.time() - mstep_begin
            
            # print("distortion:", distortion)
            if (last_distortion - distortion) < self.tolerance_:
                break
            last_distortion = distortion
        # print("estep", estep_total)
        # print("mstep", mstep_total)
        # print("fit", time.time() - fit_begin)
        # 屏蔽结束

    def predict(self, p_datas, method = 0):
        result = []
        # 作业2
        # 屏蔽开始
        
        predict_begin = time.time()
        if method == 0:
            # Brute froce
            for p_data in p_datas:
                # get distances of p_data to all centers
                # dists is a length-k array
                dists = np.linalg.norm(self.centers_-p_data, axis=1)
                result.append(np.argmin(dists))
        elif method == 1:
            # KD Tree
            pass

        # print("predict", time.time()-predict_begin)
        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

