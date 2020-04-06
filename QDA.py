#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:42:06 2019

@author: macbookthibaultlahire
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import colors


def QDARegression(X, y):
    
    data_label1 = X[y==1.0]
    data_label0 = X[y==0.0]

    pi = np.sum(y==1.0) * 1.0 / len(X)
    
    mu1 = np.zeros(2)
    mu0 = np.zeros(2)
    mu0 = X[y==0.].mean(axis=0)
    mu1 = X[y==1.].mean(axis=0)
    
    X1 = X[y==1] - mu1
    x1x1 = np.sum((X1[:,0])**2)
    x2x2 = np.sum((X1[:,1])**2)
    x1x2 = np.sum(X1[:,0] * X1[:,1])

    sigma1 = np.zeros((2,2))
    l1 = len(data_label1)
    sigma1[0][0] = x1x1/l1
    sigma1[0][1] = x1x2/l1
    sigma1[1][0] = x1x2/l1
    sigma1[1][1] = x2x2/l1
    
    X0 = X[y==0] - mu0
    x1x1 = np.sum((X0[:,0])**2)
    x2x2 = np.sum((X0[:,1])**2)
    x1x2 = np.sum(X0[:,0] * X0[:,1])
    
    sigma0 = np.zeros((2,2))
    l0 = len(data_label0)
    sigma0[0][0] = x1x1/l0
    sigma0[0][1] = x1x2/l0
    sigma0[1][0] = x1x2/l0
    sigma0[1][1] = x2x2/l0
    
    return pi, mu0, mu1, sigma0, sigma1



def predict_QDARegression(X, pi, mu0, mu1, sigma0, sigma1):
    n, p = X.shape
#    X2 = np.concatenate((X, np.ones((n, 1))), axis=1)
    sigma1_inv = np.linalg.inv(sigma1)
    sigma0_inv = np.linalg.inv(sigma0)
    pred = np.zeros(n)
    factor = ((np.sqrt(np.linalg.det(sigma1)))/(np.sqrt(np.linalg.det(sigma0)))) * ((1-pi)/pi)
    for i in range(n):
        xi = np.zeros(2)
        xi[0] = X[i][0]
        xi[1] = X[i][1]
        tmp = -0.5*(xi-mu0).T.dot(sigma0_inv).dot(xi-mu0) + 0.5* (xi-mu1).T.dot(sigma1_inv).dot(xi-mu1)
        pred[i] = 1.0/(1+factor*np.exp(tmp))    
    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0
    return pred



def layout_QDARegression(X_trainA, y_trainA, X_trainB, y_trainB, X_trainC, y_trainC, X_testA, y_testA, X_testB, y_testB, X_testC, y_testC):
    # apply QDA for all training data
    pi_A, mu0_A, mu1_A, sigma0_A, sigma1_A = QDARegression(X_trainA, y_trainA)
    pi_B, mu0_B, mu1_B, sigma0_B, sigma1_B = QDARegression(X_trainB, y_trainB)
    pi_C, mu0_C, mu1_C, sigma0_C, sigma1_C = QDARegression(X_trainC, y_trainC)
    
    
    # Colormap
    cmap = colors.LinearSegmentedColormap(
        'red_blue_classes',
        {'blue': [(0, 0.9, 0.9), (1, 1, 1)],
         'green': [(0, 0.9, 0.9), (1, 0.9, 0.9)],
         'red': [(0, 1, 1), (1, 0.9, 0.9)]})
    plt.cm.register_cmap(cmap=cmap)
    # class 0 and 1 : areas 
    nx, ny = 200, 100
    x_min = 5
    x_max = 17
    y_min = 0
    y_max = 15
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # to get the predict probability
    predict_proba_A = predict_QDARegression(grid, pi_A, mu0_A, mu1_A, sigma0_A, sigma1_A)
    predict_proba_A = predict_proba_A.reshape(xx.shape)
    predict_proba_B = predict_QDARegression(grid, pi_B, mu0_B, mu1_B, sigma0_B, sigma1_B)
    predict_proba_B = predict_proba_B.reshape(xx.shape)
    predict_proba_C = predict_QDARegression(grid, pi_C, mu0_C, mu1_C, sigma0_C, sigma1_C)
    predict_proba_C = predict_proba_C.reshape(xx.shape)
    
    # Represent graphically the data as well as the conic defined by p(y = 1|x) = 0.5
    fig, ax = plt.subplots(2, 3, figsize=(25, 16))
    
    ax[0,0].pcolormesh(xx, yy, predict_proba_A, cmap='red_blue_classes')
    ax[0,0].contour(xx, yy, predict_proba_A, [0.5])
    ax[0,0].scatter(X_trainA[y_trainA==1.0][:,0], X_trainA[y_trainA==1.0][:,1], color="blue", alpha=0.5, label = "Class 1")
    ax[0,0].scatter(X_trainA[y_trainA==0.0][:,0], X_trainA[y_trainA==0.0][:,1], color="red", alpha=0.5, label = "Class 0")
    ax[0,0].legend() 
    ax[0,0].set_xlabel('$x_1$')
    ax[0,0].set_ylabel('$x_2$')
    ax[0,0].set_title('Dataset A : training data')
    
    ax[1,0].pcolormesh(xx, yy, predict_proba_A, cmap='red_blue_classes')
    ax[1,0].contour(xx, yy, predict_proba_A, [0.5])
    ax[1,0].scatter(X_testA[y_testA==1.0][:,0], X_testA[y_testA==1.0][:,1], color="blue", alpha=0.5, label = "Class 1")
    ax[1,0].scatter(X_testA[y_testA==0.0][:,0], X_testA[y_testA==0.0][:,1], color="red", alpha=0.5, label = "Class 0")
    ax[1,0].legend() 
    ax[1,0].set_xlabel('$x_1$')
    ax[1,0].set_ylabel('$x_2$')
    ax[1,0].set_title('Dataset A : test data')
    
    ax[0,1].pcolormesh(xx, yy, predict_proba_B, cmap='red_blue_classes')
    ax[0,1].contour(xx, yy, predict_proba_B, [0.5])
    ax[0,1].scatter(X_trainB[y_trainB==1.0][:,0], X_trainB[y_trainB==1.0][:,1], color="blue", alpha=0.5, label = "Class 1")
    ax[0,1].scatter(X_trainB[y_trainB==0.0][:,0], X_trainB[y_trainB==0.0][:,1], color="red", alpha=0.5, label = "Class 0")
    ax[0,1].legend()
    ax[0,1].set_xlabel('$x_1$')
    ax[0,1].set_ylabel('$x_2$') 
    ax[0,1].set_title('Dataset B : training data')
    
    ax[1,1].pcolormesh(xx, yy, predict_proba_B, cmap='red_blue_classes')
    ax[1,1].contour(xx, yy, predict_proba_B, [0.5])
    ax[1,1].scatter(X_testB[y_testB==1.0][:,0], X_testB[y_testB==1.0][:,1], color="blue", alpha=0.5, label = "Class 1")
    ax[1,1].scatter(X_testB[y_testB==0.0][:,0], X_testB[y_testB==0.0][:,1], color="red", alpha=0.5, label = "Class 0")
    ax[1,1].legend()
    ax[1,1].set_xlabel('$x_1$')
    ax[1,1].set_ylabel('$x_2$')
    ax[1,1].set_title('Dataset B : test data')
    
    ax[0,2].pcolormesh(xx, yy, predict_proba_C, cmap='red_blue_classes')
    ax[0,2].contour(xx, yy, predict_proba_C, [0.5])
    ax[0,2].scatter(X_trainC[y_trainC==1.0][:,0], X_trainC[y_trainC==1.0][:,1], color="blue", alpha=0.5, label = "Class 0")
    ax[0,2].scatter(X_trainC[y_trainC==0.0][:,0], X_trainC[y_trainC==0.0][:,1], color="red", alpha=0.5, label = "Class 1")
    ax[0,2].legend()
    ax[0,2].set_xlabel('$x_1$')
    ax[0,2].set_ylabel('$x_2$') 
    ax[0,2].set_title('Dataset C : training data')
    
    ax[1,2].pcolormesh(xx, yy, predict_proba_C, cmap='red_blue_classes')
    ax[1,2].contour(xx, yy, predict_proba_C, [0.5])
    ax[1,2].scatter(X_testC[y_testC==1.0][:,0], X_testC[y_testC==1.0][:,1], color="blue", alpha=0.5, label = "Class 1")
    ax[1,2].scatter(X_testC[y_testC==0.0][:,0], X_testC[y_testC==0.0][:,1], color="red", alpha=0.5, label = "Class 0")
    ax[1,2].legend() 
    ax[1,2].set_xlabel('$x_1$')
    ax[1,2].set_ylabel('$x_2$')
    ax[1,2].set_title('Dataset C : test data')
    
    fig.savefig("ResQDAReg")
    fig.tight_layout



def mis(pred, true):
    return np.sum(np.abs(pred-true))/pred.shape[0]


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
    
    layout_QDARegression(X_trainA, y_trainA, X_trainB, y_trainB, X_trainC, y_trainC, X_testA, y_testA, X_testB, y_testB, X_testC, y_testC)

    print("ClassificationA :\n")
    
    pi_A, mu0_A, mu1_A, sigma0_A, sigma1_A = QDARegression(X_trainA, y_trainA)
    print("pi_A =", pi_A)
    print("mu0_A = ", mu0_A)
    print("mu1_A = ", mu1_A)
    print("sigma0_A =", sigma0_A)
    print("sigma1_A =\n", sigma1_A)
    pred = predict_QDARegression(X_trainA, pi_A, mu0_A, mu1_A, sigma0_A, sigma1_A)
    print("Error on trainA:", mis(pred, y_trainA))
    pred = predict_QDARegression(X_testA, pi_A, mu0_A, mu1_A, sigma0_A, sigma1_A)
    print("Error on testA:", mis(pred, y_testA))
    
    print("\nClassificationB :\n")
    
    pi_B, mu0_B, mu1_B, sigma0_B, sigma1_B = QDARegression(X_trainB, y_trainB)
    print("pi_B =", pi_B)
    print("mu0_B = ", mu0_B)
    print("mu1_B = ", mu1_B)
    print("sigma0_B =", sigma0_B)
    print("sigma1_B =\n", sigma1_B)
    pred = predict_QDARegression(X_trainB, pi_B, mu0_B, mu1_B, sigma0_B, sigma1_B)
    print("Error on trainB:", mis(pred, y_trainB))
    pred = predict_QDARegression(X_testB, pi_B, mu0_B, mu1_B, sigma0_B, sigma1_B)
    print("Error on testB:", mis(pred, y_testB))
    
    print("\nClassificationC :\n")
    
    pi_C, mu0_C, mu1_C, sigma0_C, sigma1_C = QDARegression(X_trainC, y_trainC)
    print("pi_C =", pi_C)
    print("mu0_C = ", mu0_C)
    print("mu1_C = ", mu1_C)
    print("sigma0_C =", sigma0_C)
    print("sigma1_C =\n", sigma1_C)
    pred = predict_QDARegression(X_trainC, pi_C, mu0_C, mu1_C, sigma0_C, sigma1_C)
    print("Error on trainC:", mis(pred, y_trainC))
    pred = predict_QDARegression(X_testC, pi_C, mu0_C, mu1_C, sigma0_C, sigma1_C)
    print("Error on testC:", mis(pred, y_testC))

