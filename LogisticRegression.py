import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import scipy.io as scio

def poly_transformation(X,n):
    N = X.shape[0]
    X_poly = np.matrix(np.ones((N, X.shape[1]+1)))
    X_poly[:,1:3] = X
    return X_poly

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cost_function(X,y,theta,ld):
    m = X.shape[0]
    h = sigmoid(X * theta)
    t1 = np.log(h).T * y
    t2 = np.log(1-h).T * (1 - y)
    t3 = ld / (2 * m) * np.sum(np.square(theta[1:]))
    cost = -(1/float(m)) * np.sum(t1 + t2) + t3
    #print(np.sum(t1 + t2))
    #print(-(1/float(m)))
    return cost

def gradient(X,y,theta,ld):
    m = X.shape[0]
    h = sigmoid(X*theta)
    grad = (1/float(m)) * X.T * (h-y)
    grad[1:] = grad[1:] + (ld/float(m)) * theta[1:]
    return grad

def gradientDescent(X,y,theta,alpha,mini,iters,ld):
    n = len(X)
    cost = np.zeros(iters)
    for j in range(iters):
        for mini_batch in mini:
            X_tilde = mini_batch[0]
            y_tilde = mini_batch[1]
            gradient_J = gradient(X_tilde,y_tilde,theta,ld)
            temp = theta - ((alpha/len(X_tilde)) * gradient_J)
            
            theta = temp
            cost[j] = cost_function(X,y,theta,ld)
        
    #if j % 50 == 0:
    #    print("Loss iter",j,": ",cost[j])
        
    return theta,cost

def theta_initialization(X):
    theta = np.matrix(np.zeros((X.shape[1],1)))
    return theta

def mini_batch(X,y,m):
    random_indices = np.random.choice(len(X),len(y),replace=False)
    X_new = X[random_indices,:]
    y_new = y[random_indices]
    mini = []
    for i in range(0,len(y),m):
        mini.append([X_new[i:i+m,:],y_new[i:i+m]])
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
        
def k_fold_CV_GD(X,y,k,theta,alpha,mini,iters):
    lmd = [0.2,0.3,0.4,0.5,0.1,1,0.0006,.00008]
    E_ho = []
    for l in lmd:
        E = []
        for X_v, y_v, X_t,y_t in k_fold(X,y,k):
            #compute theta
            theta,cost = gradientDescent(X_t,y_t,theta,alpha,mini,iters,l)
            #print(theta.shape)
            E.append(np.power(np.linalg.norm(y_v - (X_v * theta)),2))
            #compute holdout error
        #compute sum(holdout error)
        #print('For lambda',l)
        #print('cost size',cost.shape)
        #print('Cost',cost[999])
        E_ho.append(np.mean(E))
        
    #pick lambda
    lmd_new = np.argmin(E_ho)
    #lmbd_new = argmin(sum(error))
    return theta,lmd[lmd_new]

def predict(theta, X):
    m = X.shape[0]
    p = np.matrix(np.zeros((X.shape[0],1)))
    h = sigmoid(X*theta)
    
    for i in range(m):
        if h[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p

def prediction_error(p,y):
    count = 0
    for i in range(len(y)):
        if p[i] == y[i]:
            count += 1
    return 1 - count / float(len(y))

def logisticRegression():
	data = scio.loadmat('HW2_Data/data1.mat')
	X = data['X_trn']
	y = data['Y_trn']
	X = np.matrix(X)
	y = np.matrix(y)
	X_tst = data['X_tst']
	y_tst = data['Y_tst']
	X_tst = np.matrix(X_tst)
	y_tst = np.matrix(y_tst)
	#Extract and plot dat
	pos = np.where(y == 1)[0]
	neg = np.where(y == 0)[0]

	plt.scatter([X[pos,0]],[X[pos,1]],color='black', marker='+', label='y=1' )
	plt.scatter([X[neg,0]],[X[neg,1]],color='yellow', marker='o', label='y=0' )
	plt.xlabel('Test 1')
	plt.ylabel('Test 2')
	plt.legend(['Class1','Class2'])
	plt.show()

	data2 = scio.loadmat('HW2_Data/data2.mat')
	X2 = data2['X_trn']
	y2 = data2['Y_trn']
	X2 = np.matrix(X2)
	y2 = np.matrix(y2)
	X_tst2 = data['X_tst']
	y_tst2 = data['Y_tst']
	X_tst2 = np.matrix(X_tst2)
	y_tst2 = np.matrix(y_tst2)
	#Extract and plot dat
	pos = np.where(y2 == 1)[0]
	neg = np.where(y2 == 0)[0]

	plt.scatter([X2[pos,0]],[X2[pos,1]],color='black', marker='+', label='y=1' )
	plt.scatter([X2[neg,0]],[X2[neg,1]],color='yellow', marker='o', label='y=0' )
	plt.xlabel('Test 1')
	plt.ylabel('Test 2')
	plt.legend(['Class1','Class2'])
	plt.show()


	X_train = poly_transformation(X,1)
	X_test = poly_transformation(X_tst,1)
	theta = theta_initialization(X_train)
	X_train2 = poly_transformation(X2,1)
	X_test2 = poly_transformation(X_tst2,1)
	theta2 = theta_initialization(X_train2)
	mini = mini_batch(X_train,y,17)
	t_sd,l_sd = k_fold_CV_GD(X_train,y,2,theta,0.1,mini,1000)
	#print('###################################')
	p = predict(t_sd, X_test)
	p.shape[0]
	#Extract and plot dat
	x1 = np.array([X[:,1].min(), X[:,1].max()])
	x2 = - t_sd.item(0) / t_sd.item(2) + x1 * (- t_sd.item(1) / t_sd.item(2))
	#print x1
	#print x2
	# Plot decision boundary
	plt.plot(x1, x2, color='k', ls='--', lw=2)

	pos = np.where(y == 1)[0]
	neg = np.where(y == 0)[0]

	plt.scatter([X[pos,0]],[X[pos,1]],color='black', marker='+', label='y=1' )
	plt.scatter([X[neg,0]],[X[neg,1]],color='yellow', marker='o', label='y=0' )
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend(['Decision Boundary','Class1','Class2'])
	plt.show()

	p = predict(t_sd, X_train)
	e1 = prediction_error(p,y)
	print e1

	mini2 = mini_batch(X_train2,y2,17)
	t_sd2,l_sd2 = k_fold_CV_GD(X_train2,y2,2,theta2,0.01,mini2,1000)
	p2 = predict(t_sd2, X_test2)
	#Extract and plot dat
	x1_2 = np.array([X_train2[:,1].min(), X_train2[:,1].max()])
	x2_2 = - t_sd2.item(0) / t_sd2.item(2) + x1_2 * (- t_sd2.item(1) / t_sd2.item(2))

	# Plot decision boundary
	plt.plot(x1_2, x2_2, color='k', ls='--', lw=2)

	pos = np.where(y2 == 1)[0]
	neg = np.where(y2 == 0)[0]

	plt.scatter([X2[pos,0]],[X2[pos,1]],color='black', marker='+', label='y=1' )
	plt.scatter([X2[neg,0]],[X2[neg,1]],color='yellow', marker='o', label='y=0' )
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend(['Decision Boundary','Class1','Class2'])
	plt.show()

	pt = predict(t_sd2, X_train2)
	err = prediction_error(pt,y2)
	print err

def main():
    logisticRegression()

if __name__ == '__main__':
    main()