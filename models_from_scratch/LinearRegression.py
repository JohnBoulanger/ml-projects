"""
Linear Regression from Scratch (Closed-Form Solution)

Author: John Boulanger
Date: Feb 1st 2026

Description:
This module implements linear regression using the normal equation
(closed-form solution) without relying on high-level machine learning
libraries for model fitting. A bias term is explicitly added to the
feature matrix, and weights are computed using the Moore–Penrose
pseudoinverse.

The implementation is intended for educational purposes, illustrating
the mathematical foundations of linear regression, including:
- Explicit bias handling
- Weight computation via (XᵀX)⁻¹Xᵀy using np.linalg.pinv
- Mean squared error evaluation

This code is suitable for small to medium datasets and serves as a
reference implementation rather than a production-optimized solution.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    # add bias vector of ones to training set
    bias = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((bias, X_train))

    # compute closed form solution w = (psuedo inverse of X) * y
    w = np.linalg.pinv(X_train) @ y_train

    return w

def mse(X_train,y_train,w):
    # get prediction
    y_hat = pred(X_train, w)

    # calculate mse
    errors = y_hat - y_train
    n = X_train.shape[0]
    avgError = np.sum(errors ** 2) / n

    return avgError
    

def pred(X_train,w):
    # our prediction y_hat on the training set is simply X_train @ w
    bias = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((bias, X_train))
    return X_train @ w

def test_SciKit(X_train, X_test, Y_train, Y_test):
    # instantiate model object
    lr = linear_model.LinearRegression()
    # fit to training set
    lr.fit(X_train, Y_train)
    # predict using model on test set
    y_pred = lr.predict(X_test)
    # calculate mse
    error = mean_squared_error(Y_test, y_pred)

    return error

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def test_LR():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------test_LR-------------------')
test_LR()

# The performance of my implementation is effectively identical to scikit-learn’s
# LinearRegression, as both produce the same mean squared error as during one run I got:
# 2926.414591386882 (2a) vs 2926.414591386884 (2b). Differences are likely due to
# floating-point numerical precision. This indicates that the closed-form solution 
# matches scikit-learn’s implementation.
