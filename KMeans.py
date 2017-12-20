import numpy as np
import math
from random import sample
from copy import deepcopy

'''
Initialize the centroids
'''
def initializeCentroids(Y,k):
    r = range(0,Y.shape[0])
    print Y.shape[0]
    rand_indices = sample(range(0,Y.shape[0]),k)
    #print rand_indices
    return np.array([Y[i] for i in rand_indices])

'''
Returns the distance between two data points
'''
def dist(x,y):
    return np.linalg.norm(x-y)


'''
Input: Dataset (take transform of the dataset depending your given data)
d is # of clusters, iters is the # of iterations, we can use convergence
condition here instead of iteration loop
Returns the clusters 
'''
def kMeans(Y,d=10,iters=100):
    centroids = initializeCentroids(Y.T, d)
    C_old = np.zeros(centroids.shape)
    clusters = np.zeros(Y.T.shape[0])
    indxs = np.zeros(Y.T.shape[0])
    error = dist(centroids,C_old)
    while iters != 0:
        for i in range(Y.T.shape[0]):
            dis = 9999999999999
            for j in xrange(d):
                distance = dist(Y.T[i],centroids[j])
                if dis > distance:
                    cluster = j
                    dis = distance
            clusters[i] = cluster
        C_old = deepcopy(centroids)
        for i in range(d):
            points = [Y.T[j] for j in range(Y.T.shape[0]) if clusters[j]==i]
            centroids[i] = np.mean(points, axis=0)
        iters -=1
    return clusters
