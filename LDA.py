#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:52:49 2019

@author: macbookthibaultlahire
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle



def LDARegression(X, y):
    
    data_label1 = X[y==1]
    data_label0 = X[y==0]

    pi = np.sum(y==1)/len(X)
    
    mu1 = np.zeros(2)
    mu0 = np.zeros(2)
    mu0 = X[y==0].mean(axis=0)
    mu1 = X[y==1].mean(axis=0)
    
    
    X1 = X[y==1] - mu1
    x1x1 = np.sum((X1[:,0])**2)
    x2x2 = np.sum((X1[:,1])**2)
    x1x2 = np.sum(X1[:,0] * X1[:,1])

    l1 = len(data_label1)
    sigma1 = np.zeros((2,2))
    sigma1[0][0] = x1x1/l1
    sigma1[0][1] = x1x2/l1
    sigma1[1][0] = x1x2/l1
    sigma1[1][1] = x2x2/l1
    
    
    X0 = X[y==0] - mu0
    x1x1 = np.sum((X0[:,0])**2)
    x2x2 = np.sum((X0[:,1])**2)
    x1x2 = np.sum(X0[:,0] * X0[:,1])
    
    l0 = len(data_label0)
    sigma0 = np.zeros((2,2))
    sigma0[0][0] = x1x1/l0
    sigma0[0][1] = x1x2/l0
    sigma0[1][0] = x1x2/l0
    sigma0[1][1] = x2x2/l0
    sigma = np.zeros((2,2))
    sigma = pi * sigma1 + (1-pi) * sigma0

    
    sigma_inv = np.linalg.inv(sigma)
    w = sigma_inv.dot(mu1 - mu0)
    w = w.tolist()
    b = 0.5 * mu0.T.dot(sigma_inv).dot(mu0) - 0.5 * mu1.T.dot(sigma_inv).dot(mu1) 
    b = [b]
    W = np.asarray(w + b)
    
    return W, pi, mu0, mu1, sigma

    


def predict_LDARegression(w, pi, X):
    n = X.shape[0]
    # to include the bias term in w :
    X2 = np.concatenate((X, np.ones((n, 1))), axis=1)
    pred = 1./(1+(1-pi)/pi*np.exp(-(X2.dot(w))))
    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0
    return pred


def mis(pred, true):
    return np.sum(np.abs(pred-true))/pred.shape[0]


def layout_LDARegression(X_trainA, y_trainA, X_trainB, y_trainB, X_trainC, y_trainC, X_testA, y_testA, X_testB, y_testB, X_testC, y_testC):
    w_A, piA, mu0A, mu1A, sigmaA = LDARegression(X_trainA, y_trainA)
    w_B, piB, mu0B, mu1B, sigmaB = LDARegression(X_trainB, y_trainB)
    w_C, piC, mu0C, mu1C, sigmaC = LDARegression(X_trainC, y_trainC)
    
    fig, ax = plt.subplots(2,3, figsize=(25,16))
    x = np.array([6, 16])

    ax[0,0].scatter(X_trainA[y_trainA==1.0][:,0], X_trainA[y_trainA==1.0][:,1], color="blue", alpha=0.5)
    ax[0,0].scatter(X_trainA[y_trainA==0.0][:,0], X_trainA[y_trainA==0.0][:,1], color="red", alpha=0.5)
    y = (-w_A[2]-w_A[0]*x + np.log(piA/(1-piA))) / w_A[1]
    ax[0,0].plot(x, y)
    ax[0,0].set_title("Dataset A : training data")
    ax[0,0].legend(("LDAReg","Class 1","Class 0"))
    ax[0,0].set_xlabel("$x_1$")
    ax[0,0].set_ylabel("$x_2$")
    
    ax[0,1].scatter(X_trainB[y_trainB==1.0][:,0], X_trainB[y_trainB==1.0][:,1], color="blue", alpha=0.5)
    ax[0,1].scatter(X_trainB[y_trainB==0.0][:,0], X_trainB[y_trainB==0.0][:,1], color="red", alpha=0.5)
    y = (-w_B[2]-w_B[0]*x + np.log(piB/(1-piB))) / w_B[1]
    ax[0,1].plot(x,y)
    ax[0,1].set_title("Dataset B : training data")
    ax[0,1].legend(("LDAReg","Class 1","Class 0"))
    ax[0,1].set_xlabel("$x_1$")
    ax[0,1].set_ylabel("$x_2$")
    
    ax[0,2].scatter(X_trainC[y_trainC==1.0][:,0], X_trainC[y_trainC==1.0][:,1], color="blue", alpha=0.5)
    ax[0,2].scatter(X_trainC[y_trainC==0.0][:,0], X_trainC[y_trainC==0.0][:,1], color="red", alpha=0.5)
    y = (-w_C[2]-w_C[0]*x + np.log(piC/(1-piC))) / w_C[1]
    ax[0,2].plot(x,y)
    ax[0,2].set_title("Dataset C : training data")
    ax[0,2].legend(("LDAReg","Class 1","Class 0"))
    ax[0,2].set_xlabel("$x_1$")
    ax[0,2].set_ylabel("$x_2$")
    
    ax[1,0].scatter(X_testA[y_testA==1.0][:,0], X_testA[y_testA==1.0][:,1], color="blue", alpha=0.5)
    ax[1,0].scatter(X_testA[y_testA==0.0][:,0], X_testA[y_testA==0.0][:,1], color="red", alpha=0.5)
    y = (-w_A[2]-w_A[0]*x + np.log(piA/(1-piA))) / w_A[1]
    ax[1,0].plot(x, y)
    ax[1,0].set_title("Dataset A : test data")
    ax[1,0].legend(("LDAReg","Class 1","Class 0"))
    ax[1,0].set_xlabel("$x_1$")
    ax[1,0].set_ylabel("$x_2$")
    
    ax[1,1].scatter(X_testB[y_testB==1.0][:,0], X_testB[y_testB==1.0][:,1], color="blue", alpha=0.5)
    ax[1,1].scatter(X_testB[y_testB==0.0][:,0], X_testB[y_testB==0.0][:,1], color="red", alpha=0.5)
    y = (-w_B[2]-w_B[0]*x + np.log(piB/(1-piB))) / w_B[1]
    ax[1,1].plot(x,y)
    ax[1,1].set_title("Dataset B : test data")
    ax[1,1].legend(("LDAReg","Class 1","Class 0"))
    ax[1,1].set_xlabel("$x_1$")
    ax[1,1].set_ylabel("$x_2$")
    
    ax[1,2].scatter(X_testC[y_testC==1.0][:,0], X_testC[y_testC==1.0][:,1], color="blue", alpha=0.5)
    ax[1,2].scatter(X_testC[y_testC==0.0][:,0], X_testC[y_testC==0.0][:,1], color="red", alpha=0.5)
    y = (-w_C[2]-w_C[0]*x + np.log(piC/(1-piC))) / w_C[1]
    ax[1,2].plot(x,y)
    ax[1,2].set_title("Dataset C : test data")
    ax[1,2].legend(("LDAReg","Class 1","Class 0"))
    ax[1,2].set_xlabel("$x_1$")
    ax[1,2].set_ylabel("$x_2$")
    
    fig.savefig("ResLDAReg")
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
    
    layout_LDARegression(X_trainA, y_trainA, X_trainB, y_trainB, X_trainC, y_trainC, X_testA, y_testA, X_testB, y_testB, X_testC, y_testC)

    
    w_A, pi_A, mu0_A, mu1_A, sigma_A = LDARegression(X_trainA, y_trainA)
    w_B, pi_B, mu0_B, mu1_B, sigma_B = LDARegression(X_trainB, y_trainB)
    w_C, pi_C, mu0_C, mu1_C, sigma_C = LDARegression(X_trainC, y_trainC)

    print("w_A = ", w_A/w_A[1])
    print("w_B = ", w_B/w_B[1])
    print("w_C = ", w_C/w_C[1])
    
    print("\nClassificationA :\n")
    
    print("pi_A =", pi_A)
    print("mu0_A = ", mu0_A)
    print("mu1_A = ", mu1_A)
    print("sigma_A =", sigma_A)
    pred = predict_LDARegression(w_A, pi_A, X_trainA)
    print("Error on trainA:", mis(pred, y_trainA))
    pred = predict_LDARegression(w_A, pi_A, X_testA)
    print("Error on testA:", mis(pred, y_testA))
    
    print("\nClassificationB :\n")
    
    print("pi_B =", pi_B)
    print("mu0_B = ", mu0_B)
    print("mu1_B = ", mu1_B)
    print("sigma_B =", sigma_B)
    pred = predict_LDARegression(w_B, pi_B, X_trainB)
    print("Error on trainB:", mis(pred, y_trainB))
    pred = predict_LDARegression(w_B, pi_B, X_testB)
    print("Error on testB:", mis(pred, y_testB))
    
    print("\nClassificationC :\n")
    
    print("pi_C =", pi_C)
    print("mu0_C = ", mu0_C)
    print("mu1_C = ", mu1_C)
    print("sigma_C =", sigma_C)
    pred = predict_LDARegression(w_C, pi_C, X_trainC)
    print("Error on trainC:", mis(pred, y_trainC))
    pred = predict_LDARegression(w_C, pi_C, X_testC)
    print("Error on testC:", mis(pred, y_testC))

