import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.io as scio


def computeTheta(X,y):
    theta = np.linalg.pinv(X.T * X) * X.T * y    
    return theta

def computeCost(X,y,theta):
    m = 2 * len(X) # length of training set
    inner = np.power(((X * theta.T) - y), 2)
    cost = np.sum(inner) / m
    return cost


def MSE(y, y_pred):
    return np.mean(np.power((y - y_pred),2))

def mini_batch(X,y,m):
    random_indices = np.random.choice(len(X),len(y),replace=False)
    X_new = X[random_indices,:]
    y_new = y[random_indices]
    mini = []
    for i in range(0,len(y),m):
        mini.append([X[i:i+m,:],y[i:i+m]])
    return mini

def stochasticGD(X,y,theta,alpha,mini,iters):
    n = len(X)
    #temp = np.matrix(np.zeros(theta.shape))
    #parameters = theta.ravel().shape[1]
    # while convergence condition
    cost = np.zeros(iters)
    for j in range(iters):
        for mini_batch in mini:
            X_tilde = mini_batch[0]
            y_tilde = mini_batch[1]
            gradient_J = X_tilde.T * ((X_tilde * theta) - y_tilde)
            temp = theta - ((alpha / len(X_tilde)) * gradient_J)
            
            theta = temp
            #cost[j] = computeCost(X_tilde, y_tilde, theta)
        
        #if j % 50 == 0:
        #    print("Loss iter",j,": ",cost[j])
        
    return theta

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

def linearRegression():
    data = scio.loadmat('HW1_Data/dataset1.mat')
    X = data['X_trn']
    y = data['Y_trn']

    X = np.matrix(X)
    y = np.matrix(y)
    '''
    print('Enter degree: ')
    n = raw_input()
    '''
    X_tst = data['X_tst']
    y_tst = data['Y_tst']
    X_tst = np.matrix(X_tst)
    y_tst = np.matrix(y_tst)

    '''
    ##############################################
    ############Closed Form Solution##############
    ##############################################
    For degree 1:
    '''
    X_train = poly_transformation(X,1)
    X_test = poly_transformation(X_tst,1)

    w_1 = computeTheta(X_train, y)
    y_train_predict_1 = np.dot(X_train, w_1)
    mses_squared = MSE(y, y_train_predict_1)
    print('Training error for degree 1: ', mses_squared)
    y_test_predict_1 = np.dot(X_test, w_1)
    mses_test_1 = MSE(y_tst, y_test_predict_1)
    print('Testing error for degree 1: ', mses_test_1)
    
    '''
    For degree 2:
    '''
    X_train_2 = poly_transformation(X,2)
    X_test_2 = poly_transformation(X_tst,2)


    w_2 = computeTheta(X_train_2, y,)
    y_train_predict_2 = np.dot(X_train_2, w_2)
    mses_2 = MSE(y, y_train_predict_2)
    print('Training error for degree 2: ', mses_2)
    y_test_predict_2 = np.dot(X_test_2, w_2)
    mses_test_2 = MSE(y_tst, y_test_predict_2)
    print('Testing error for degree 2: ', mses_test_2)
    
    '''
    For degree 3:
    '''
    X_train_3 = poly_transformation(X,3)
    X_test_3 = poly_transformation(X_tst,3)


    w_3 = computeTheta(X_train_3, y)
    y_train_predict_3 = np.dot(X_train_3, w_3)
    mses_3 = MSE(y, y_train_predict_3)
    print('Training error for degree 3: ', mses_3)
    y_test_predict_3 = np.dot(X_test_3, w_3)
    mses_test_3 = MSE(y_tst, y_test_predict_3)
    print('Testing error for degree 3: ', mses_test_3)
    
    '''
    For degree 3:
    '''
    X_train_5 = poly_transformation(X,5)
    X_test_5 = poly_transformation(X_tst,5)


    w_5 = computeTheta(X_train_5, y)
    y_train_predict_5 = np.dot(X_train_5, w_5)
    mses_5 = MSE(y, y_train_predict_5)
    print('Training error for degree 5: ', mses_5)
    y_test_predict_5 = np.dot(X_test_5, w_5)
    mses_test_5 = MSE(y_tst, y_test_predict_5)
    print('Testing error for degree 3: ', mses_test_5)
    
    '''
    #########################################
    #############Gradient Descent############
    #########################################
    For degree 2:
    '''
    alpha = 0.00001
    iters = 1000
    theta2 = theta_initialization(X_train_2)
    mini = mini_batch(X_train_2,y,5)
    
    w_2_sgd = stochasticGD(X_train_2, y,theta2,alpha,mini,iters)
    y_train_predict_2_sgd = np.dot(X_train_2, w_2_sgd)
    mses_2_sgd = MSE(y, y_train_predict_2_sgd)
    print('Training error for degree 2 for GD: ', mses_2_sgd)
    y_test_predict_2_sgd = np.dot(X_test_2, w_2_sgd)
    mses_test_2_sgd = MSE(y_tst, y_test_predict_2_sgd)
    print('Testing error for degree 2 for GD: ', mses_test_2_sgd)
    
    '''
    For degree 3:
    '''
    theta3 = theta_initialization(X_train_3)
    mini3 = mini_batch(X_train_3,y,5)
    w_3_sgd = stochasticGD(X_train_3, y,theta3,alpha,mini3,iters)
    y_train_predict_3_sgd = np.dot(X_train_3, w_3_sgd)
    mses_3_sgd = MSE(y, y_train_predict_3_sgd)
    print('Training error for degree 3 for GD: ', mses_3_sgd)
    y_test_predict_3_sgd = np.dot(X_test_3, w_3_sgd)
    mses_test_3_sgd = MSE(y_tst, y_test_predict_3_sgd)
    print('Testing error for degree 3 for GD: ', mses_test_3_sgd)
    
    '''
    For degree 5:
    '''
    theta5 = theta_initialization(X_train_5)
    mini5 = mini_batch(X_train_5,y,5)
    w_5_sgd = stochasticGD(X_train_5, y,theta5,alpha,mini5,iters)
    y_train_predict_5_sgd = np.dot(X_train_5, w_5_sgd)
    mses_5_sgd = MSE(y, y_train_predict_5_sgd)
    print('Training error for degree 5 for GD: ', mses_5_sgd)
    y_test_predict_5_sgd = np.dot(X_test_5, w_5_sgd)
    mses_test_5_sgd = MSE(y_tst, y_test_predict_5_sgd)
    print('Testing error for degree 5 for GD: ', mses_test_5_sgd)

def main():
    linearRegression()

if __name__ == '__main__':
    main()