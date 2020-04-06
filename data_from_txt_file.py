#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:55:09 2019

@author: macbookthibaultlahire
"""

import numpy as np
import pickle


file = open('data/testA','r', encoding='ASCII')
lines = file.readlines()

n = len(lines)
d = 2

A = np.zeros((n,d))
y = np.zeros(n)


for i in range(n):
    aux = lines[i].split()    
    y[i] = int(aux[2][1])
    for j in range(d):
        A[i,j] = aux[j]



pickle.dump(A,open('./data/X_testA','wb'))
pickle.dump(y,open('./data/y_testA','wb'))
