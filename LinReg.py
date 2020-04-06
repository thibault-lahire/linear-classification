#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:05:00 2019

@author: macbookthibaultlahire
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


def linearRegression(X, y):
    n = X.shape[0]
    # to include the bias term in w :
    X2 = np.concatenate((X, np.ones((n, 1))), axis=1)
    w = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y)
    return w


def predict_linearRegression(w, X): 
    n = X.shape[0]
    X2 = np.concatenate((X, np.ones((n, 1))), axis=1)
    pred = X2.dot(w)
    pred[pred>0.5] = 1
    pred[pred <= 0.5] = 0
    return pred

def layout_linearRegression(X_trainA, y_trainA, X_trainB, y_trainB, X_trainC, y_trainC, X_testA, y_testA, X_testB, y_testB, X_testC, y_testC):
    w_A = linearRegression(X_trainA, y_trainA)
    w_B = linearRegression(X_trainB, y_trainB)
    w_C = linearRegression(X_trainC, y_trainC)
        
    fig, ax = plt.subplots(2,3, figsize=(25,16))
    x = np.array([6, 16])

    ax[0,0].scatter(X_trainA[y_trainA==1.0][:,0], X_trainA[y_trainA==1.0][:,1], color="blue", alpha=0.5)
    ax[0,0].scatter(X_trainA[y_trainA==0.0][:,0], X_trainA[y_trainA==0.0][:,1], color="red", alpha=0.5)
    y = (-w_A[2]-w_A[0]*x + 0.5) / w_A[1]
    ax[0,0].plot(x, y)
    ax[0,0].set_title("Dataset A : training data")
    ax[0,0].legend(("LinReg","Class 1","Class 0"))
    ax[0,0].set_xlabel("$x_1$")
    ax[0,0].set_ylabel("$x_2$")
    
    ax[0,1].scatter(X_trainB[y_trainB==1.0][:,0], X_trainB[y_trainB==1.0][:,1], color="blue", alpha=0.5)
    ax[0,1].scatter(X_trainB[y_trainB==0.0][:,0], X_trainB[y_trainB==0.0][:,1], color="red", alpha=0.5)
    y = (-w_B[2]-w_B[0]*x + 0.5) / w_B[1]
    ax[0,1].plot(x,y)
    ax[0,1].set_title("Dataset B : training data")
    ax[0,1].legend(("LinReg","Class 1","Class 0"))
    ax[0,1].set_xlabel("$x_1$")
    ax[0,1].set_ylabel("$x_2$")
    
    ax[0,2].scatter(X_trainC[y_trainC==1.0][:,0], X_trainC[y_trainC==1.0][:,1], color="blue", alpha=0.5)
    ax[0,2].scatter(X_trainC[y_trainC==0.0][:,0], X_trainC[y_trainC==0.0][:,1], color="red", alpha=0.5)
    y = (-w_C[2]-w_C[0]*x + 0.5) / w_C[1]
    ax[0,2].plot(x,y)
    ax[0,2].set_title("Dataset C : training data")
    ax[0,2].legend(("LinReg","Class 1","Class 0"))
    ax[0,2].set_xlabel("$x_1$")
    ax[0,2].set_ylabel("$x_2$")
    
    ax[1,0].scatter(X_testA[y_testA==1.0][:,0], X_testA[y_testA==1.0][:,1], color="blue", alpha=0.5)
    ax[1,0].scatter(X_testA[y_testA==0.0][:,0], X_testA[y_testA==0.0][:,1], color="red", alpha=0.5)
    y = (-w_A[2]-w_A[0]*x + 0.5) / w_A[1]
    ax[1,0].plot(x, y)
    ax[1,0].set_title("Dataset A : test data")
    ax[1,0].legend(("LinReg","Class 1","Class 0"))
    ax[1,0].set_xlabel("$x_1$")
    ax[1,0].set_ylabel("$x_2$")
    
    ax[1,1].scatter(X_testB[y_testB==1.0][:,0], X_testB[y_testB==1.0][:,1], color="blue", alpha=0.5)
    ax[1,1].scatter(X_testB[y_testB==0.0][:,0], X_testB[y_testB==0.0][:,1], color="red", alpha=0.5)
    y = (-w_B[2]-w_B[0]*x + 0.5) / w_B[1]
    ax[1,1].plot(x,y)
    ax[1,1].set_title("Dataset B : test data")
    ax[1,1].legend(("LinReg","Class 1","Class 0"))
    ax[1,1].set_xlabel("$x_1$")
    ax[1,1].set_ylabel("$x_2$")
    
    ax[1,2].scatter(X_testC[y_testC==1.0][:,0], X_testC[y_testC==1.0][:,1], color="blue", alpha=0.5)
    ax[1,2].scatter(X_testC[y_testC==0.0][:,0], X_testC[y_testC==0.0][:,1], color="red", alpha=0.5)
    y = (-w_C[2]-w_C[0]*x + 0.5) / w_C[1]
    ax[1,2].plot(x,y)
    ax[1,2].set_title("Dataset C : test data")
    ax[1,2].legend(("LinReg","Class 1","Class 0"))
    ax[1,2].set_xlabel("$x_1$")
    ax[1,2].set_ylabel("$x_2$")
    
    fig.savefig("ResLinReg")
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
    
    layout_linearRegression(X_trainA, y_trainA, X_trainB, y_trainB, X_trainC, y_trainC, X_testA, y_testA, X_testB, y_testB, X_testC, y_testC)

    w_A = linearRegression(X_trainA, y_trainA)
    w_B = linearRegression(X_trainB, y_trainB)
    w_C = linearRegression(X_trainC, y_trainC)
    
    print("w_A = ", w_A/w_A[1])
    print("w_B = ", w_B/w_B[1])
    print("w_C = ", w_C/w_C[1])
    
    
    


