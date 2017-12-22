%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.io as scio
from random import sample
from copy import deepcopy
from skimage.transform import resize

def softmax(z):
    def softmax(z):
    dom = np.sum(np.exp(z))
    #print dom
    #print z.shape[0]
    a3 = np.array([np.exp(z[i,0])/dom for i in range(z.shape[0])]).reshape(10,1)
    return a3

def dsoftmax(z):
    return softmax(z)*(1 - softmax(z))

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def dsigmoid(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def hyperbolic(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

def dhyperbolic(z):
    return 1 - (hyperbolic(z)**2)

def relu(z):
    return max(0,z)

def drelu(z):
    if z > 0:
        return 1
    else:
        return 0



def forwardPropogation(X,w_hidden,b_input,w_output,b_hidden,S1,S2,S3):
    a_input = np.zeros((X.shape))
    a_hidden = np.zeros((S2,1))
    #print a_hidden.shape
    a_output = np.zeros((S3,1))
    #print X.shape
    for i in range(X.shape[0]):
        a_input[i] = X[i] 
    
    
    z_hidden = w_hidden.dot(a_input.T) + b_input
    #print z_hidden.shape
    a_hidden = hyperbolic(z_hidden)
    '''
    for i in range(S2):
        temp = 0.0
        for j in range(S1):
            temp += a_input[j] * w_hidden[i][j]
        a_hidden[i] = temp + b_input[i]
        z_hidden[i] = hyperbolic(a_hidden[i])
    '''
    #print w_output.shape
    #print a_hidden.shape
    z_output = w_output.dot(a_hidden) + b_hidden
    #print z_output.shape
    #print a_output.shape
    a_output = hyperbolic(z_output)
    #print z_output.shape
    '''
    for i in range(S3):
        temp = 0.0
        for j in range(S2):
            temp += a_hidden[j] * w_output[i][j]
        a_output[i] = temp + b_hidden[i]
        z_output[i] = softmax(a_output[i])
    '''
    #for layer in NN:
    #    for neuron in layer:
    #        a = activation(w,x)
    #        z = w * a + b
    return a_input, a_hidden, a_output, z_hidden, z_output

def calculateY(a):
    y = []
    for i in range(a.shape[0]):
        y.append(softmax(a[i]))
    return np.array(y)

def neuralNetworks(data,S1,S2,S3,iterations):
    # initialize z and b
    lR = 1
    ld = 0.01
    w_hidden = np.zeros((S2,S1))
    b_input = np.zeros((S2,1))
    w_output = np.zeros((S3,S2))
    b_hidden = np.zeros((S3,1))
    
    for x in np.nditer(w_hidden, op_flags=['readwrite']):
        x[...] = np.random.normal(0,0.5)
    #for x in np.nditer(b_input, op_flags=['readwrite']):       
    #    x[...] = np.random.normal(0,0.01)    
    for x in np.nditer(w_output, op_flags=['readwrite']):
        x[...] = np.random.normal(0,0.5)
    #for x in np.nditer(b_hidden, op_flags=['readwrite']):       
    #    x[...] = np.random.normal(0,0.01)
        
    w2 = w_hidden
    w3 = w_output
    b2 = b_input
    b3 = b_hidden
    #print w2.shape
    #print w3.shape
    for iters in xrange(iterations):
        a1, a2, a3, z2, z3 = forwardPropogation(data,w2,b2,w3,b3,S1,S2,S3)
        #print a2.shape # transpose
        #print a3.shape
        y = softmax(a3)
        #print y.shape
        ds_z3 = (1 - a3) * a3
        #print ds_z3.shape
        del_output = (a3 - y)* a3#derivative of softmax
        #print del_output.shape
        tmp = del_output.T.dot(w3)
        #ds_z2 = (1-a2) * a2
        ds_z2 = dhyperbolic(a2)
        #print tmp.shape
        #print ds_z2.shape
        del_hidden = tmp * ds_z2.T
        #print del_hidden.shape
        
        '''
        def wtf(_data, _del):
            del_w = np.zeros((_del.shape[0],_data.shape[0]))
            print del_w.shape
            del_b = np.zeros((_del.shape[0],1))
            for i in xrange(_data.shape[0]):
                #J_w, J_b = backPropogation()
                dJ_w = _del *_data[i]
                dJ_b = _del
                print _del.shape
                print _data[i].shape
                print dJ_w.shape
                del_w = del_w + dJ_w
                del_b = del_b + dJ_b
            return del_w/float(_data.shape[0]), del_b/float(_data.shape[0])
        '''
        
        #dw3, db3 = wtf(a2, del_output)
        #dw2, db2 = wtf(a1, del_hidden)
        dw3 = del_output.dot(a2.T)
        dw2 = del_hidden.T.dot(a1)
        db3 = np.sum(del_output, axis=1, keepdims=True)
        db2 = np.sum(del_hidden.T, axis=1, keepdims=True)
        
        w3 = w3 - lR * (dw3 + (ld * w3))
        b3 = b3 - lR * db3
        w2 = w2 - lR * (dw2 + (ld * w2))
        b2 = b2 - lR * db2
        #print y
    return w2, w3, b2, b3, a3

