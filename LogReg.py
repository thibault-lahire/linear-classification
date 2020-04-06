#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 21:53:33 2019

@author: macbookthibaultlahire
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


def sigmoid(x):
    return 1./(1.+np.exp(-x))


def likelihood(w, X, y):
    return y.dot(np.log(sigmoid(X.dot(w)))) + (1-y).dot(np.log(sigmoid(-X.dot(w))))



def logisticRegression(X, y):
    n, p = X.shape
    X2 = np.concatenate((X, np.ones((n, 1))), axis=1)
    # the bias term is included in w
    w = np.zeros(p+1)
    
    eta = sigmoid(X2.dot(w))
    grad = X2.T.dot(y-eta)
    hessian = - X2.T.dot(np.diag(eta*(1-eta))).dot(X2)
    direction = - np.linalg.inv(hessian).dot(grad)
    stop_criterion = 1
    
    step_size = 1.0
    a = 0.5 # decreasing rate
    while stop_criterion > 1e-15:
        #line search
        while likelihood(w + step_size * direction, X2, y) < likelihood(w, X2, y) + 0.5 * step_size * grad.dot(direction):
                step_size *= a
        
        w = w + step_size * direction
        
        eta = sigmoid(X2.dot(w))
        grad = X2.T.dot(y-eta)
        hessian = - X2.T.dot(np.diag(eta*(1-eta))).dot(X2)
        direction = - np.linalg.inv(hessian).dot(grad)
        stop_criterion = np.dot(grad, - direction) / 2 
        
    return w



def predict_logisticRegression(w, X): 
    n, p = X.shape
    X2 = np.concatenate((X, np.ones((n, 1))), axis=1)
    pred = sigmoid(X2.dot(w))
    pred[pred>0.5] = 1
    pred[pred <= 0.5] = 0
    return pred


def layout_logisticRegression(X_trainA, y_trainA, X_trainB, y_trainB, X_trainC, y_trainC, X_testA, y_testA, X_testB, y_testB, X_testC, y_testC):
    w_A = logisticRegression(X_trainA, y_trainA)
    w_B = logisticRegression(X_trainB, y_trainB)
    w_C = logisticRegression(X_trainC, y_trainC)
    
    
    fig, ax = plt.subplots(2,3, figsize=(25,16))
    x = np.array([6, 16])

    ax[0,0].scatter(X_trainA[y_trainA==1.0][:,0], X_trainA[y_trainA==1.0][:,1], color="blue", alpha=0.5)
    ax[0,0].scatter(X_trainA[y_trainA==0.0][:,0], X_trainA[y_trainA==0.0][:,1], color="red", alpha=0.5)
    y = (-w_A[2]-w_A[0]*x) / w_A[1]
    ax[0,0].plot(x, y)
    ax[0,0].set_title("Dataset A : training data")
    ax[0,0].legend(("LogReg","Class 1","Class 0"))
    ax[0,0].set_xlabel("$x_1$")
    ax[0,0].set_ylabel("$x_2$")
    
    ax[0,1].scatter(X_trainB[y_trainB==1.0][:,0], X_trainB[y_trainB==1.0][:,1], color="blue", alpha=0.5)
    ax[0,1].scatter(X_trainB[y_trainB==0.0][:,0], X_trainB[y_trainB==0.0][:,1], color="red", alpha=0.5)
    y = (-w_B[2]-w_B[0]*x) / w_B[1]
    ax[0,1].plot(x,y)
    ax[0,1].set_title("Dataset B : training data")
    ax[0,1].legend(("LogReg","Class 1","Class 0"))
    ax[0,1].set_xlabel("$x_1$")
    ax[0,1].set_ylabel("$x_2$")
    
    ax[0,2].scatter(X_trainC[y_trainC==1.0][:,0], X_trainC[y_trainC==1.0][:,1], color="blue", alpha=0.5)
    ax[0,2].scatter(X_trainC[y_trainC==0.0][:,0], X_trainC[y_trainC==0.0][:,1], color="red", alpha=0.5)
    y = (-w_C[2]-w_C[0]*x) / w_C[1]
    ax[0,2].plot(x,y)
    ax[0,2].set_title("Dataset C : training data")
    ax[0,2].legend(("LogReg","Class 1","Class 0"))
    ax[0,2].set_xlabel("$x_1$")
    ax[0,2].set_ylabel("$x_2$")
    
    ax[1,0].scatter(X_testA[y_testA==1.0][:,0], X_testA[y_testA==1.0][:,1], color="blue", alpha=0.5)
    ax[1,0].scatter(X_testA[y_testA==0.0][:,0], X_testA[y_testA==0.0][:,1], color="red", alpha=0.5)
    y = (-w_A[2]-w_A[0]*x) / w_A[1]
    ax[1,0].plot(x, y)
    ax[1,0].set_title("Dataset A : test data")
    ax[1,0].legend(("LogReg","Class 1","Class 0"))
    ax[1,0].set_xlabel("$x_1$")
    ax[1,0].set_ylabel("$x_2$")
    
    ax[1,1].scatter(X_testB[y_testB==1.0][:,0], X_testB[y_testB==1.0][:,1], color="blue", alpha=0.5)
    ax[1,1].scatter(X_testB[y_testB==0.0][:,0], X_testB[y_testB==0.0][:,1], color="red", alpha=0.5)
    y = (-w_B[2]-w_B[0]*x) / w_B[1]
    ax[1,1].plot(x,y)
    ax[1,1].set_title("Dataset B : test data")
    ax[1,1].legend(("LogReg","Class 1","Class 0"))
    ax[1,1].set_xlabel("$x_1$")
    ax[1,1].set_ylabel("$x_2$")
    
    ax[1,2].scatter(X_testC[y_testC==1.0][:,0], X_testC[y_testC==1.0][:,1], color="blue", alpha=0.5)
    ax[1,2].scatter(X_testC[y_testC==0.0][:,0], X_testC[y_testC==0.0][:,1], color="red", alpha=0.5)
    y = (-w_C[2]-w_C[0]*x) / w_C[1]
    ax[1,2].plot(x,y)
    ax[1,2].set_title("Dataset C : test data")
    ax[1,2].legend(("LogReg","Class 1","Class 0"))
    ax[1,2].set_xlabel("$x_1$")
    ax[1,2].set_ylabel("$x_2$")
    
    fig.savefig("ResLogReg")
    fig.tight_layout()



if __name__ == '__main__':
    X_trainA = pickle.load(open('data/X_trainA', 'rb'))
    y_trainA = pickle.load(open('data/y_trainA', 'rb'))
    X_trainB = pickle.load(open('data/X_trainB', 'rb'))
    y_trainB = pickle.load(open('data/y_trainB', 'rb'))
    X_trainC = pickle.load(open('data/X_trainC', 'rb'))
    y_trainC = pickle.load(open('data/y_trainC', 'rb'))
    
    X_testA = pickle.load(open('data/X_testA', 'rb'))
    y_testA = pickle.load(open('data/y_testA', 'rb'))
    X_testB = pickle.load(open('data/X_testB', 'rb'))
    y_testB = pickle.load(open('data/y_testB', 'rb'))
    X_testC = pickle.load(open('data/X_testC', 'rb'))
    y_testC = pickle.load(open('data/y_testC', 'rb'))
    
    layout_logisticRegression(X_trainA, y_trainA, X_trainB, y_trainB, X_trainC, y_trainC, X_testA, y_testA, X_testB, y_testB, X_testC, y_testC)
    
    w_A = logisticRegression(X_trainA, y_trainA)
    w_B = logisticRegression(X_trainB, y_trainB)
    w_C = logisticRegression(X_trainC, y_trainC)
        
    print("w_A = ", w_A/w_A[1])
    print("w_B = ", w_B/w_B[1])
    print("w_C = ", w_C/w_C[1])

