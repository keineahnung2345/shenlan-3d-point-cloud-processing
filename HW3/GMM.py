# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from tqdm import tqdm
plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, dim = 2, max_iter=50, tolerance=0.001):
        # how to choose tolerance value?
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.dim = dim
        
        # 屏蔽开始
        # 更新W
        
        # 更新pi
        self.weights = np.ones(n_clusters)/n_clusters
        
        # 更新Mu
        self.means = np.random.random((n_clusters, self.dim))
        
        # 更新Var
        """
        GMM.py:45: RuntimeWarning: invalid value encountered in double_scalars
          (1/pow(np.linalg.det(self.covs[j]), 0.5)) * \
        numpy\linalg\linalg.py:2116: RuntimeWarning: invalid value encountered in det
          r = _umath_linalg.det(a, signature=signature)
        """
        # self.covs = np.random.random((n_clusters, self.dim, self.dim))
        self.covs = np.array(n_clusters * [np.identity(self.dim)])
        
        self.tolerance = tolerance 
        
        # print("weights", self.weights)
        # print("means", self.means)
        # print("covs", self.covs)
        # print("covs det", np.linalg.det(self.covs[0]))
        # print("covs det sqrt", pow(np.linalg.det(self.covs[0]), 0.5))
        
        # 屏蔽结束
    
    def _gauss(self, j, datum):
        # j: the id of gaussian model
        # datum: we need to calculate the prob of datum in this model
        # print("j", j)
        # print("cov", self.covs[j])
        # print("det", np.linalg.det(self.covs[j]))
        # print("inv", np.linalg.inv(self.covs[j]))
        return 1/pow(2*np.pi, self.dim/2) * \
            (1/pow(np.linalg.det(self.covs[j]), 0.5)) * \
            np.exp(-1/2*np.dot(np.dot((datum-self.means[j]).T, 
                               np.linalg.inv(self.covs[j])),
                               (datum-self.means[j])))
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        
        N = data.shape[0]
        last_log_likelihood = float("-inf")
        
        for cur_iter in range(self.max_iter): 
            # print("iter", cur_iter)
            # E-step: calculate posterior probability
            
            # posterior probability
            post_probs = np.zeros((N, self.n_clusters))
            
            for i, datum in enumerate(data):
                for j in range(self.n_clusters):
                    # pdf_official = multivariate_normal.pdf(datum, 
                    #     mean=self.means[j], cov=self.covs[j])
                    # print("pdf official:", pdf_official)
                    # print("pdf self:", self._gauss(j, datum))
                    # assert(np.allclose(pdf_official, self._gauss(j, datum)))
                    post_probs[i][j] = self.weights[j]*self._gauss(j, datum)
                
                post_probs[i] /= post_probs[i].sum()
    
            # M-step: update weights, means and covs
            for j in range(self.n_clusters):
                N_j = post_probs[:,j].sum()
                # view post_probs[:,j] as vector and data as matrix
                # calculate their dot product
                
                # method 1
                self.means[j] = np.zeros(self.dim)
                for i, datum in enumerate(data):
                    self.means[j] += post_probs[i][j] * datum
                self.means[j] /= N_j
                
                # method 2
                #self.means[j] = post_probs[:,j].dot(data) / N_j
                
                self.covs[j] = np.zeros((self.dim, self.dim))
                for i in range(N):
                    diff = np.array([data[i] - self.means[j]])
                    # print(diff.dot(diff.T))
                    # print(np.matmul(diff.T, diff))
                    self.covs[j] += post_probs[i][j] * \
                        np.matmul(diff.T, diff)
                        # (data[i] - self.means[j]).dot((data[i] - self.means[j]).T)
                self.covs[j] /= N_j
                
                self.weights[j] = N_j/N
                
            log_likelihood = 0
            for i in range(N):
                tmp = 0
                for j in range(self.n_clusters):
                    tmp += self.weights[j] * self._gauss(j, data[i])
                log_likelihood += np.log(tmp)
            
            # print(cur_iter, "'s log likelihood:", log_likelihood)
            # if log_likelihood - last_log_likelihood < self.tolerance:
            #     break
            last_log_likelihood = log_likelihood
        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        N = data.shape[0]
        
        post_probs = np.zeros((N, self.n_clusters))
            
        for i, datum in enumerate(data):
            for j in range(self.n_clusters):
                post_probs[i][j] = self.weights[j]*self._gauss(j, datum)
            
            post_probs[i] /= post_probs[i].sum()
        
        return np.argmax(post_probs, axis=1)
        # 屏蔽结束

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

