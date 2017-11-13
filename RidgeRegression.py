import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import scipy.io as scio

def MSE(y, y_pred):
    return np.mean(np.power((y - y_pred),2))

def poly_transformation(X,n):
    N = X.shape[0]
    X_poly = np.matrix(np.ones((N,n+1)))
    #print (X_poly.shape)
    for i in range(1,n+1):
        X_poly[:,i] = np.power(X[:,0],i)
    return X_poly

def theta_initialization(X):
    theta = np.matrix(np.zeros((X.shape[1],1)))
    return theta

def mini_batch(X,y,m):
    random_indices = np.random.choice(len(X),len(y),replace=False)
    X_new = X[random_indices,:]
    y_new = y[random_indices]
    mini = []
    for i in range(0,len(y),m):
        mini.append([X[i:i+m,:],y[i:i+m]])
    return mini

def k_fold(X,y,k):
    n = X.shape[0]
    indx = list(range(0, n))
    #[0.......n]
    size = int(math.ceil(n/k))
    #size of fold

    for i in range(k):
        val_indx = range(i*size,min(size*(i+1), n))
        train_indx = np.delete(indx,val_indx)

        yield X[val_indx], y[val_indx], X[train_indx],y[train_indx]

def ridgeRegression(X,y,lbd):
    # identity matrix
    var = np.dot(X.T, X) + np.eye(X.shape[1]) * lbd
    cov = np.dot(X.T, y)
    w = np.dot(np.linalg.inv(var), cov)
    return w

def RidgeGD(X,y,theta,alpha,mini,iters,lmd):
    n = len(X)
    #temp = np.matrix(np.zeros(theta.shape))
    #parameters = theta.ravel().shape[1]
    # while convergence condition
    cost = np.zeros(iters)
    for j in range(iters):
        for mini_batch in mini:
            X_tilde = mini_batch[0]
            y_tilde = mini_batch[1]
            gradient_J = 2 * X_tilde.T * ((X_tilde * theta) - y_tilde) + (2 * np.dot(lmd,theta))
            temp = theta - ((alpha / len(X_tilde)) * gradient_J)
            
            theta = temp
            #cost[j] = computeCost(X_tilde, y_tilde, theta)
        
        #if j % 50 == 0:
        #    print("Loss iter",j,": ",cost[j])
        
    return theta

def k_fold_CV_GD(X,y,k,theta,alpha,mini,iters):
    lmd = [0.2,0.3,0.4,0.5,0.1]
    E_ho = []
    # compute theta
    #p = X.shape[0]/k
    
    for l in lmd:
        E = []
        for X_v, y_v, X_t,y_t in k_fold(X,y,k):
            #compute theta
            theta = RidgeGD(X_t,y_t,theta,alpha,mini,iters,l)
            E.append(np.power(np.linalg.norm(y_v - (X_v * theta)),2))
            #compute holdout error
        #compute sum(holdout error)
        E_ho.append(np.mean(E))
    #pick lambda
    lmd_new = np.argmin(E_ho)
    #lmbd_new = argmin(sum(error))
    return theta,lmd[lmd_new]

def k_fold_CV(X,y,k):
    lmd = [0.2,0.3,0.4,0.5,0.1]
    E_ho = []
    # compute theta
    #p = X.shape[0]/k
    
    for l in lmd:
        E = []
        for X_v, y_v, X_t,y_t in k_fold(X,y,k):
            #compute theta
            theta = ridgeRegression(X_t,y_t,l)
            E.append(np.power(np.linalg.norm(y_v - (X_v * theta)),2))
            #compute holdout error
        #compute sum(holdout error)
        E_ho.append(np.mean(E))
    #pick lambda
    lmd_new = np.argmin(E_ho)
    #lmbd_new = argmin(sum(error))
    return theta,lmd[lmd_new]

def RidgeRegression():
    data2 = scio.loadmat('dataset2.mat')
    X = data2['X_trn']
    y = data2['Y_trn']

    X = np.matrix(X)
    y = np.matrix(y)
    '''
    print('Enter degree: ')
    n = raw_input()
    '''
    X_tst = data2['X_tst']
    y_tst = data2['Y_tst']
    X_tst = np.matrix(X_tst)
    y_tst = np.matrix(y_tst)
    
    
    '''
    For degree 2:
    '''
    X_train_2 = poly_transformation(X,2)
    X_test_2 = poly_transformation(X_tst,2)
    mini_2 = mini_batch(X_train_2,y,24)

    t_2_2,l_2_2 = k_fold_CV(X_train_2,y,2)

    w_interaction_reg_2_2 = ridgeRegression(X_train_2, y,l_2_2)

    y_train_predict_interaction_reg_2_2 = np.dot(X_train_2, w_interaction_reg_2_2)
    y_test_predict_interaction_reg_2_2 = np.dot(X_test_2, w_interaction_reg_2_2)
    print('with regularization with degree 2 fold 2, MSE of train set is %.4f, MSE of test set is %.4f' % 
      (MSE(y, y_train_predict_interaction_reg_2_2), MSE(y_tst, y_test_predict_interaction_reg_2_2)))
    
    '''
    For degree 3:
    '''
    X_train_3 = poly_transformation(X,3)
    X_test_3 = poly_transformation(X_tst,3)
    mini_3 = mini_batch(X_train_3,y,24)

    t_3_2,l_3_2 = k_fold_CV(X_train_3,y,2)

    w_interaction_reg_3_2 = ridgeRegression(X_train_3, y,l_3_2)

    y_train_predict_interaction_reg_3_2 = np.dot(X_train_3, w_interaction_reg_3_2)
    y_test_predict_interaction_reg_3_2 = np.dot(X_test_3, w_interaction_reg_3_2)
    print('with regularization degree 3 fold 2, MSE of train set is %.4f, MSE of test set is %.4f' % 
      (MSE(y, y_train_predict_interaction_reg_3_2), MSE(y_tst, y_test_predict_interaction_reg_3_2)))
    
    '''
    For degree 5:
    '''
    X_train_5 = poly_transformation(X,5)
    X_test_5 = poly_transformation(X_tst,5)
    mini_5 = mini_batch(X_train_5,y,24)

    t_5_2,l_5_2 = k_fold_CV(X_train_5,y,2)

    w_interaction_reg_5_2 = ridgeRegression(X_train_5, y,l_5_2)

    y_train_predict_interaction_reg_5_2 = np.dot(X_train_5, w_interaction_reg_5_2)
    y_test_predict_interaction_reg_5_2 = np.dot(X_test_5, w_interaction_reg_5_2)
    print('with regularization degree 5 fold 2, MSE of train set is %.4f, MSE of test set is %.4f' % 
      (MSE(y, y_train_predict_interaction_reg_5_2), MSE(y_tst, y_test_predict_interaction_reg_5_2)))


    '''
    For degree 2:
    '''
    

    t_2_10,l_2_10 = k_fold_CV(X_train_2,y,10)

    w_interaction_reg_2_10 = ridgeRegression(X_train_2, y,l_2_10)

    y_train_predict_interaction_reg_2_10 = np.dot(X_train_2, w_interaction_reg_2_10)
    y_test_predict_interaction_reg_2_10 = np.dot(X_test_2, w_interaction_reg_2_10)
    print('with regularization with degree 2 fold 10, MSE of train set is %.4f, MSE of test set is %.4f' % 
      (MSE(y, y_train_predict_interaction_reg_2_10), MSE(y_tst, y_test_predict_interaction_reg_2_10)))
    
    '''
    For degree 3:
    '''
    

    t_3_10,l_3_10 = k_fold_CV(X_train_3,y,10)

    w_interaction_reg_3_10 = ridgeRegression(X_train_3, y,l_3_10)

    y_train_predict_interaction_reg_3_10 = np.dot(X_train_3, w_interaction_reg_3_10)
    y_test_predict_interaction_reg_3_10 = np.dot(X_test_3, w_interaction_reg_3_10)
    print('with regularization degree 3 fold 10, MSE of train set is %.4f, MSE of test set is %.4f' % 
      (MSE(y, y_train_predict_interaction_reg_3_10), MSE(y_tst, y_test_predict_interaction_reg_3_10)))
    
    '''
    For degree 5:
    '''
    

    t_5_10,l_5_10 = k_fold_CV(X_train_5,y,10)

    w_interaction_reg_5_10 = ridgeRegression(X_train_5, y,l_5_10)

    y_train_predict_interaction_reg_5_10 = np.dot(X_train_5, w_interaction_reg_5_10)
    y_test_predict_interaction_reg_5_10 = np.dot(X_test_5, w_interaction_reg_5_10)
    print('with regularization degree 5 fold 10, MSE of train set is %.4f, MSE of test set is %.4f' % 
      (MSE(y, y_train_predict_interaction_reg_5_10), MSE(y_tst, y_test_predict_interaction_reg_5_10)))
    
    '''
    For degree 2:
    '''
    

    t,l = k_fold_CV(X_train_2,y,X.shape[0])

    w_interaction_reg = ridgeRegression(X_train_2, y,l)

    y_train_predict_interaction_reg = np.dot(X_train_2, w_interaction_reg)
    y_test_predict_interaction_reg = np.dot(X_test_2, w_interaction_reg)
    print('with regularization with degree 2 fold N, MSE of train set is %.4f, MSE of test set is %.4f' % 
      (MSE(y, y_train_predict_interaction_reg), MSE(y_tst, y_test_predict_interaction_reg)))
    
    '''
    For degree 3:
    '''
    

    t3,l3 = k_fold_CV(X_train_3,y,X.shape[0])

    w_interaction_reg3 = ridgeRegression(X_train_3, y,l3)

    y_train_predict_interaction_reg3 = np.dot(X_train_3, w_interaction_reg3)
    y_test_predict_interaction_reg3 = np.dot(X_test_3, w_interaction_reg3)
    print('with regularization degree 3 fold N, MSE of train set is %.4f, MSE of test set is %.4f' % 
      (MSE(y, y_train_predict_interaction_reg3), MSE(y_tst, y_test_predict_interaction_reg3)))
    
    '''
    For degree 5:
    '''
    
    t5,l5 = k_fold_CV(X_train_5,y,X.shape[0])

    w_interaction_reg5 = ridgeRegression(X_train_5, y,l5)

    y_train_predict_interaction_reg5 = np.dot(X_train_5, w_interaction_reg5)
    y_test_predict_interaction_reg5 = np.dot(X_test_5, w_interaction_reg5)
    print('with regularization degree 5 fold N, MSE of train set is %.4f, MSE of test set is %.4f' % 
      (MSE(y, y_train_predict_interaction_reg5), MSE(y_tst, y_test_predict_interaction_reg5)))
    
    '''
    #########################################
    #############Gradient Descent############
    #########################################
    For degree 2:
    '''
    #list1 = [2,3,5]
    #list2 = [2,10,X.shape[0]]
    
    alpha = 0.001
    iters = 1000
    
    theta_2 = theta_initialization(X_train_2)
    
    t_sd,l_sd = k_fold_CV_GD(X_train_2,y,2,theta_2,alpha,mini_2,iters)

    w_interaction_reg = RidgeGD(X_train_2,y,t_sd,alpha,mini_2,iters,l_sd)

    y_train_predict_interaction_reg = np.dot(X_train_2, w_interaction_reg)
    y_test_predict_interaction_reg = np.dot(X_test_2, w_interaction_reg)
    print('with regularization SD with degree 2 fold 2, MSE of train set is %.4f, MSE of test set is %.4f' % 
          (MSE(y, y_train_predict_interaction_reg), MSE(y_tst, y_test_predict_interaction_reg)))
    
       
    t_sd_10,l_sd_10 = k_fold_CV_GD(X_train_2,y,10,theta_2,alpha,mini_2,iters)

    w_interaction_reg_10 = RidgeGD(X_train_2,y,theta_2,alpha,mini_2,iters,l_sd_10)

    y_train_predict_interaction_reg_10 = np.dot(X_train_2, w_interaction_reg_10)
    y_test_predict_interaction_reg_10 = np.dot(X_test_2, w_interaction_reg_10)
    print('with regularization SD with degree 2 fold 10, MSE of train set is %.4f, MSE of test set is %.4f' % 
          (MSE(y, y_train_predict_interaction_reg_10), MSE(y_tst, y_test_predict_interaction_reg_10)))
        
        
    t_sd_N,l_sd_N = k_fold_CV_GD(X_train_2,y,X.shape[0],theta_2,alpha,mini_2,iters)

    w_interaction_reg_N = RidgeGD(X_train_2,y,theta_2,alpha,mini_2,iters,l_sd_N)

    y_train_predict_interaction_reg_N = np.dot(X_train_2, w_interaction_reg_N)
    y_test_predict_interaction_reg_N = np.dot(X_test_2, w_interaction_reg_N)
    print('with regularization SD with degree 2 fold N, MSE of train set is %.4f, MSE of test set is %.4f' % 
          (MSE(y, y_train_predict_interaction_reg_N), MSE(y_tst, y_test_predict_interaction_reg_N)))
    
    '''
    For degree 3
    '''
    theta_3 = theta_initialization(X_train_3)
    
    t_sd_3,l_sd_3 = k_fold_CV_GD(X_train_3,y,2,theta_3,alpha,mini_3,iters)

    w_interaction_reg_3 = RidgeGD(X_train_3,y,t_sd_3,alpha,mini_3,iters,l_sd_3)

    y_train_predict_interaction_reg_3 = np.dot(X_train_3, w_interaction_reg_3)
    y_test_predict_interaction_reg_3 = np.dot(X_test_3, w_interaction_reg_3)
    print('with regularization SD with degree 3 fold 2, MSE of train set is %.4f, MSE of test set is %.4f' % 
          (MSE(y, y_train_predict_interaction_reg_3), MSE(y_tst, y_test_predict_interaction_reg_3)))
    
       
    t_sd_10_3,l_sd_10_3 = k_fold_CV_GD(X_train_3,y,10,theta_3,alpha,mini_3,iters)

    w_interaction_reg_10_3 = RidgeGD(X_train_3,y,theta_3,alpha,mini_3,iters,l_sd_10_3)

    y_train_predict_interaction_reg_10_3 = np.dot(X_train_3, w_interaction_reg_10_3)
    y_test_predict_interaction_reg_10_3 = np.dot(X_test_3, w_interaction_reg_10_3)
    print('with regularization SD with degree 3 fold 10, MSE of train set is %.4f, MSE of test set is %.4f' % 
          (MSE(y, y_train_predict_interaction_reg_10_3), MSE(y_tst, y_test_predict_interaction_reg_10_3)))
        
        
    t_sd_N_3,l_sd_N_3 = k_fold_CV_GD(X_train_3,y,X.shape[0],theta_3,alpha,mini_3,iters)

    w_interaction_reg_N_3 = RidgeGD(X_train_3,y,theta_3,alpha,mini_3,iters,l_sd_N_3)

    y_train_predict_interaction_reg_N_3 = np.dot(X_train_3, w_interaction_reg_N_3)
    y_test_predict_interaction_reg_N_3 = np.dot(X_test_3, w_interaction_reg_N_3)
    print('with regularization SD with degree 3 fold N, MSE of train set is %.4f, MSE of test set is %.4f' % 
          (MSE(y, y_train_predict_interaction_reg_N_3), MSE(y_tst, y_test_predict_interaction_reg_N_3)))
    
    '''
    For degree 5
    '''
    theta_5 = theta_initialization(X_train_5)
    
    t_sd_5,l_sd_5 = k_fold_CV_GD(X_train_5,y,2,theta_5,alpha,mini_5,iters)

    w_interaction_reg_5 = RidgeGD(X_train_5,y,t_sd_5,alpha,mini_5,iters,l_sd_5)

    y_train_predict_interaction_reg_5 = np.dot(X_train_5, w_interaction_reg_5)
    y_test_predict_interaction_reg_5 = np.dot(X_test_5, w_interaction_reg_5)
    print('with regularization SD with degree 5 fold 2, MSE of train set is %.4f, MSE of test set is %.4f' % 
          (MSE(y, y_train_predict_interaction_reg_5), MSE(y_tst, y_test_predict_interaction_reg_5)))
    
       
    t_sd_10_5,l_sd_10_5 = k_fold_CV_GD(X_train_5,y,10,theta_5,alpha,mini_5,iters)

    w_interaction_reg_10_5 = RidgeGD(X_train_5,y,theta_5,alpha,mini_5,iters,l_sd_10_5)

    y_train_predict_interaction_reg_10_5 = np.dot(X_train_5, w_interaction_reg_10_5)
    y_test_predict_interaction_reg_10_5 = np.dot(X_test_5, w_interaction_reg_10_5)
    print('with regularization SD with degree 5 fold 10, MSE of train set is %.4f, MSE of test set is %.4f' % 
          (MSE(y, y_train_predict_interaction_reg_10_5), MSE(y_tst, y_test_predict_interaction_reg_10_5)))
        
        
    t_sd_N_5,l_sd_N_5 = k_fold_CV_GD(X_train_5,y,X.shape[0],theta_5,alpha,mini_5,iters)

    w_interaction_reg_N_5 = RidgeGD(X_train_5,y,theta_5,alpha,mini_5,iters,l_sd_N_5)

    y_train_predict_interaction_reg_N_5 = np.dot(X_train_5, w_interaction_reg_N_5)
    y_test_predict_interaction_reg_N_5 = np.dot(X_test_5, w_interaction_reg_N_5)
    print('with regularization SD with degree 5 fold N, MSE of train set is %.4f, MSE of test set is %.4f' % 
          (MSE(y, y_train_predict_interaction_reg_N_5), MSE(y_tst, y_test_predict_interaction_reg_N_5)))

def main():
    linearRegression()

if __name__ == '__main__':
    main()