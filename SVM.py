import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import scipy.io as scio

def linearKernel(X):
    return np.dot(X, X.T)

def functionCal(alpha,y,K,b,i):
    return np.dot(K[:,i].T, np.multiply(alpha, y)) + b - y[i]

def SMO(X,y,C,tol,max_passes):
    b = 0
    m = len(X[:,0])
    #n = len(X[:,1])
    
    #y(y == 0) = -1
    
    alpha = np.matrix(np.zeros((m,1)))
    passes = 0
    K = linearKernel(X)
 
    while (passes < max_passes):
        num_changed_alphas = 0
        for i in range(0,m):
            #E[i] = functionCal(X[:,i]) - y[i]
            E_i = functionCal(alpha,y,K,b,i)
            # print (y[i] * E_i)
            
            if ((y[i] * E_i) < (- tol) and  (alpha[i] < C)) or ((y[i] * E_i) > tol and alpha[i] > 0):
                #select j != i randomly
                j = (i+1)%m
                #if j == i:
                #    break
                
                #E[j] = functionCal(X[:,j]) - y[j]
                E_j = functionCal(alpha,y,K,b,j)
                
                alpha_old_i = alpha[i].copy()
                alpha_old_j = alpha[j].copy()
                
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                elif y[i] == y[j]:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                
                if(L == H):
                    # print 'L == H'
                    continue
                
                var  = 2 * K[i,j] - K[i,i] - K[j,j]
                
                if var >= 0:
                    continue
                
                alpha[j] = alpha[j] - (y[j] * (E_i - E_j)) / var
                
                if alpha[j] > H:
                    alpha[j] = H
                elif (L <= alpha[j] and alpha[j] <= H):
                    alpha[j] = alpha[j]
                elif alpha[j] < L:
                    alpha[j] = L
                if(alpha[j] - alpha_old_j) < 1e-5:
                    
                    continue
                
                alpha[i] = alpha[i] + y[i] * y[j] * (alpha_old_j - alpha[j])
                
                b1 = b - E_i - (y[i] * np.sum(alpha[i] - alpha_old_i) * K[i,i]) \
                - (y[j] * np.sum(alpha[j] - alpha_old_j) * K[i,j])
                b2 = b - E_j - (y[i] * np.sum(alpha[i] - alpha_old_i) * K[i,j]) \
                - (y[j] * np.sum(alpha[j] - alpha_old_j) * K[j,j])
                
                if (0 < alpha[i]) and (alpha[i] < C):
                    b = b1
                elif (0 < alpha[j]) and (alpha[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas = num_changed_alphas + 1
                
            # end if
                
        # end for
        if num_changed_alphas == 0:
            passes = passes + 1
        else:
            passes = 0
    # end while
    w = np.dot(X.T, np.multiply(alpha, y))
    
    return alpha, b, w

def predict(X,w,b):
    
    m = X.shape[0]
    p = np.matrix(np.zeros((X.shape[0],1)))
    pred = np.matrix(np.zeros((X.shape[0],1)))
    
    p = X * w + b
    for i in range(m):
        if p[i] >= 0:
            pred[i] = 1
        else:
            pred[i] = 0
    return pred

def prediction_error(p,y):
    count = 0
    for i in range(len(y)):
        if p[i] == y[i]:
            count += 1
    return 1 - count / float(len(y))

def svm():
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

	_y = y.copy().astype(float)
	for i in range(len(_y)):
	    if _y[i] < 1:
	        _y[i] = -1
	    else:
	        _y[i] = 1

	a,b,w = SMO(X,_y,1,1e-5,10)

	x1 = np.array([X[:,1].min(), X[:,1].max()])
	x2 = - b.item(0) / w.item(1) + x1 * (- w.item(0) / w.item(1))
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

	_y2 = y2.copy().astype(float)
	for i in range(len(_y2)):
	    if _y2[i] < 1:
	        _y2[i] = -1
	    else:
	        _y2[i] = 1

	a2,b2,w2 = SMO(X2,_y2,1,1e-5,10)
	x1_2 = np.array([X2[:,1].min()-1, X2[:,1].max()+1])
	x2_2 = - b2.item(0) / w2.item(1) + x1_2 * (- w2.item(0) / w2.item(1))
	print x1
	print x2
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
	p = predict(X_tst2,w2,b2)
	e1 = prediction_error(p,y_tst2)
	print e1

def main():
	svm()

if __name__ == '__main__':
	main()