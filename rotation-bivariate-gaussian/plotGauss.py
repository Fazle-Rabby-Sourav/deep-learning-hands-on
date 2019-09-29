#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 22:49:32 2017

@author: maida

Plot bivariate Gaussian
"""

import numpy as np
import matplotlib.pyplot as plt

# covar matrix must be positive definite, or
# symmetric with an inverse for Gaussian distribution
covarMat = np.array([[1.0, 0.0], [0.0, 6.0]])  # 2x2 array
covarInv = np.linalg.inv(covarMat)


def gaussValue(x):  # x is 2-element column vec
    return np.exp(-0.5 * np.dot(np.dot(x.T,
                                       covarInv),
                                x))


# create a range from -6 to 6 in 0.2 increments. Length is 61.
pltRange = np.linspace(-6, 6, num=61, endpoint=True)

# array to hold the values for a gaussian function
# index z arrays by [i,j)
# indices are from 0 to 60
z1 = np.zeros((61, 61))

# for a 2D array, (0,0) is top left
# also 1st component of array index is y (vertical)

# Generate values
i = 0
for x in pltRange:
    j = 0
    for y in pltRange:
        pt1 = np.array([[x], [y]])  # column vector. 2x1 array.
        z1[j, i] = gaussValue(pt1)
        j = j + 1
    i = i + 1

z1 = np.flipud(z1)

plt.figure()
CS = plt.contour(pltRange, pltRange, z1)
plt.grid()
plt.show()
