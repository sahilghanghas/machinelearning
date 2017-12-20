import numpy as np

'''
Principal component analysis (PCA)
Input: Dataset as an array or matrix and dimension
Output: Dataset with reduced dimentionality
'''
def PCA(Y,d):
    m = np.mean(Y,axis=0)
    Y = np.subtract(Y,m)
    [U_y,S_y,V_y] = np.linalg.svd(Y)
    U = U_y[:,0:d]
    X = U.dot(np.diag(S_y)[:d,:d]).T
    return X
