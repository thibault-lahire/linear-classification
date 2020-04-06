#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:15:57 2019

@author: macbookthibaultlahire
"""

import numpy as np
import pickle

import LDA
import LogReg
import LinReg


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


def mis(pred, true):
    return np.sum(np.abs(pred-true))/pred.shape[0]

print("ClassificationA :\n")

print("LDA:")
WA, piA, mu0A, mu1A, sigmaA = LDA.LDARegression(X_trainA, y_trainA)
pred = LDA.predict_LDARegression(WA, piA, X_trainA)
print("trainA:", mis(pred, y_trainA))
pred = LDA.predict_LDARegression(WA, piA, X_testA)
print("testA:", mis(pred, y_testA))


print("\nLogistic regression:")
WA = LogReg.logisticRegression(X_trainA, y_trainA)
pred = LogReg.predict_logisticRegression(WA, X_trainA)
print("trainA:", mis(pred, y_trainA))
pred = LogReg.predict_logisticRegression(WA, X_testA)
print("testA:", mis(pred, y_testA))


# linear regression
print("\nLinear regression:")
WA = LinReg.linearRegression(X_trainA, y_trainA)
pred = LinReg.predict_linearRegression(WA, X_trainA)
print("trainA:", mis(pred, y_trainA))
pred = LinReg.predict_linearRegression(WA, X_testA)
print("testA:", mis(pred, y_testA))

print("\n")

print("ClassificationB :\n")

print("LDA:")
WB, piB, mu0B, mu1B, sigmaB = LDA.LDARegression(X_trainB, y_trainB)
pred = LDA.predict_LDARegression(WB, piB, X_trainB)
print("trainB:", mis(pred, y_trainB))
pred = LDA.predict_LDARegression(WB, piB, X_testB)
print("testB:", mis(pred, y_testB))


print("\nLogistic regression:")
WB = LogReg.logisticRegression(X_trainB, y_trainB)
pred = LogReg.predict_logisticRegression(WB, X_trainB)
print("trainB:", mis(pred, y_trainB))
pred = LogReg.predict_logisticRegression(WB, X_testB)
print("testB:", mis(pred, y_testB))


# linear regression
print("\nLinear regression:")
WB = LinReg.linearRegression(X_trainB, y_trainB)
pred = LinReg.predict_linearRegression(WB, X_trainB)
print("trainB:", mis(pred, y_trainB))
pred = LinReg.predict_linearRegression(WB, X_testB)
print("testB:", mis(pred, y_testB))

print("\n")

print("ClassificationC :\n")

print("LDA:")
WC, piC, mu0C, mu1C, sigmaC = LDA.LDARegression(X_trainC, y_trainC)
pred = LDA.predict_LDARegression(WC, piC, X_trainC)
print("trainC:", mis(pred, y_trainC))
pred = LDA.predict_LDARegression(WC, piC, X_testC)
print("testC:", mis(pred, y_testC))


print("\nLogistic regression:")
WC = LogReg.logisticRegression(X_trainC, y_trainC)
pred = LogReg.predict_logisticRegression(WC, X_trainC)
print("trainC:", mis(pred, y_trainC))
pred = LogReg.predict_logisticRegression(WC, X_testC)
print("testC:", mis(pred, y_testC))


# linear regression
print("\nLinear regression:")
WC = LinReg.linearRegression(X_trainC, y_trainC)
pred = LinReg.predict_linearRegression(WC, X_trainC)
print("trainC:", mis(pred, y_trainC))
pred = LinReg.predict_linearRegression(WC, X_testC)
print("testC:", mis(pred, y_testC))