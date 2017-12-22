import numpy as np
'''

'''
def KernelPCA(K,d):
    # k-tilde
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    k_tilde = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    val, w = np.linalg.eigh(k_tilde)
    
    ## rescale w w.r.t. lambda
    w_i = []
    for i in range(d):
        w_temp = w[:,-i-1]
        val_temp = val[-i-1]
        mean = math.sqrt(np.sum(np.square(w_temp)) * val_temp)
        w_temp = np.divide(w_temp, mean)
        w_i.append(w_temp)
    
    w_i = np.matrix(np.asarray(w_i))
    
    X = np.dot(w_i, k_tilde)
    return X
