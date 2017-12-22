import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.io as scio
from random import sample

def calculateL(Y):
    n = Y.shape[0]
    w = np.zeros((n, n))
    for i in xrange(n):
        for j in xrange(i + 1, n):
            w[i, j] = np.exp((-np.linalg.norm(Y[i] - Y[j])**2) / float(0.1))
            w[j, i] = w[i, j]
    d = np.zeros((n, n))
    for i in xrange(n):
        d[i, i] = np.sum(w[i, :])
    return d-w

def spectralClustering(Y,k,d,iters):
    L = calculateL(Y)
    w, v = np.linalg.eig(L)
    index = w.argsort()
    h = v[:, index[0]]
    for i in xrange(1, k):
        h = np.vstack((h, v[:, index[i]]))
    #h = h.T
    print h.T.shape
    labels = kMeans(h,d,100)
    return labels
