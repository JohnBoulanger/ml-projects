"""
Feedforward Neural Network from Scratch (Backpropagation + SGD)

Author: John Boulanger
Date: Feb 6th 2026

Description:
This module implements a fully connected feedforward neural network
from first principles using NumPy. The network supports an arbitrary
number of hidden layers and is trained using stochastic gradient descent
with backpropagation.

Key features:
- Manual weight initialization (including bias terms)
- Forward propagation through hidden and output layers
- Backpropagation of error gradients
- Per-epoch error tracking
- Comparison against scikit-learn's MLPClassifier for validation

The implementation emphasizes conceptual clarity over performance and
is designed to demonstrate how modern neural networks operate at a
low level, including how gradients are computed and applied.

This code is intended for learning and experimentation, not for
large-scale or production training workloads.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 


def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    # initialize return values
    err = []
    weights = []

    # input + hidden + output
    numLayers = len(hidden_layer_sizes) + 2
    # nodes per layer 
    nodes = [X_train.shape[1]] + hidden_layer_sizes + [1]

    # initialize weights using normal distribution with mean = 0 and std = 0.1
    for l in range(numLayers - 1):
        # rows = bias + nodes in layer, cols = nodes in next layer
        wl = np.random.normal(0, 0.1, (1 + nodes[l], nodes[l + 1]))
        weights.append(wl)

    # do forward + back propagation over epochs
    for epoch in range(epochs):
        error = 0.0

        # shuffle training dataset so we use randomly selected data points consistent with SGD
        shuffle_indicies = np.arange(len(X_train))
        np.random.shuffle(shuffle_indicies)
        X_train = X_train[shuffle_indicies]
        y_train = y_train[shuffle_indicies]

        # go through every training case for each epoch
        for n, xn in enumerate(X_train):
            # add bias value to training case
            bias = np.ones((1,))
            x = np.hstack((bias, xn))
            # calculate outputs at every node from forward propagation and inputs into every node aside from input layer
            X, S = forwardPropagation(x, weights)
            # calculate weight gradients using back propagation with outputs from forward propagation
            g = backPropagation(X, y_train[n], S, weights)
            # update weights using weight gradients
            weights = updateWeights(weights, g, alpha)
            # add error for this training case
            error += (errorPerSample(X, y_train[n]))

        # store error for this epoch
        err.append(error / X_train.shape[0])

    return err, weights


def forwardPropagation(x, weights):
    X = []
    S = []
    # add x (training cases) as outputs for layer 0
    X.append(x)

    for l, W in enumerate(weights):
        # S holds weighted sum of outputs from previous layer
        S.append(X[l] @ W)
        
        # Calculate activation for each neuron and store in next layer of X
        x_l = []
        for s in S[l]:
            # output layer, use outputf
            if l == len(weights) - 1:
                x_l.append(outputf(s))
            else:
                x_l.append(activation(s))

        # append new layer of inputs + bias (not for output layer though)
        if not l == len(weights) - 1:
            x_l = np.hstack(([1.0], x_l))
        X.append(x_l)

    return X, S

def errorPerSample(X,y_n):
    # X[-1][0] is the output (last layer first and only node)
    eN = errorf(X[-1][0], y_n)
    return eN

def backPropagation(X, y_n, S, weights):
    # the last output layer neuron stores our final prediction
    y_hat = X[-1][0]

    # stores derivatives of the error function wrt s
    delta = []
    # chain rule for error wrt s on output layer
    delta_output = derivativeError(y_hat, y_n) * derivativeOutput(S[-1][0])
    delta.insert(0, [delta_output])

    # apply backpropagation to get deltas for previous layers using output layer delta
    # start at the last hidden layer
    for l in reversed(range(len(S) - 1)):
        delta_l = []
        # iterate over neurons in layer l
        for j in range (len(S[l])):
            weighted_sum = 0
            # backpropagate error from layer l + 1
            for k in range(len(S[l + 1])):
                # weighted_sum += sum of weights in layer l for j neurons (skip bias) to k neurons in layer l + 1 multiplied
                # the deltas for all k neurons in layer l + 1
                weighted_sum += weights[l + 1][j + 1][k] * delta[0][k]

            # multiply by activation derivative for node j in layer l (chain rule)
            delta_l.append(weighted_sum * derivativeActivation(S[l][j]))

        # add new deltas to the front of the list
        delta.insert(0, delta_l)

    # convert deltas to weight gradients (dE / dW)
    g = []

    for l in range(len(weights)):
        gl = []
        for j in range(len(weights[l])):
            glj = []
            for k in range(len(weights[l][j])):
                glj.append(X[l][j] * delta[l][k])
            gl.append(glj)
        g.append(gl)

    return g
        

def updateWeights(weights, g, alpha):
    nW = []
    # iterate over layers
    for l in range(len(weights)):
        wl = []
        # iterate over nodes in layer l
        for j in range(len(weights[l])):
            wjk = []
            # iterate over nodes in layer l + 1
            for k in range(len(weights[l][j])):
                wjk.append(weights[l][j][k] - alpha * g[l][j][k])
            wl.append(wjk)
        nW.append(wl)

    return nW

# ReLU activation function
def activation(s):
    return max(0, s)

# derivative of ReLU
def derivativeActivation(s):
    if s > 0:
        return 1
    else:
        return 0

# sigmoid function (logisitc regression)
def outputf(s):
    x_L = 1 / (1 + np.exp(-s))
    return x_L

# derivative of sig(s) is sig(s) * (1 - sig(s))
def derivativeOutput(s):
    x_L = outputf(s) * (1 - outputf(s))
    return x_L

# log loss error function
def errorf(x_L,y):
    if y == 1:
        return -np.log(x_L)
    else:
        return -np.log(1 - x_L)

# derivative of log los error function
def derivativeError(x_L,y):
    if y == 1:
        return -1 / x_L
    else:
        return 1 / (1 - x_L)

def pred(x_n,weights):
    # get output from forward propagation
    x_n = np.hstack(([1.0], x_n))
    X, S = forwardPropagation(x_n, weights)
    # X[-1][0] is the output (last layer first and only node)
    y_hat = X[-1][0]
    c = 0
    if y_hat >= 0.5:
        c = 1
    else:
        c = -1
    return c

def confMatrix(X_train,y_train,w):
    y_hats = []
    # get all predictions for all training cases
    for x in X_train:
        y_hats.append(pred(x, w))

    # get the confusion matrix using scikit learn lib function
    cm = confusion_matrix(y_train, y_hats)
    return cm

def plotErr(e, epochs):
    # set labels
    plt.xlabel("epochs")
    plt.ylabel("log loss error")
    plt.title("log loss error vs. epochs for neural network made from scratch")
    # plot and show function e vs epochs
    plt.plot(range(epochs), e)
    plt.show()

    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    # initialize MLP Classifier with specified parameters
    nn_clf = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=(30, 10), random_state=1)
    # fit data on training set
    nn_clf.fit(X_train, Y_train)
    # predict the outputs using a test set
    y_hats = nn_clf.predict(X_test)
    # get the confusion matrix 
    cm = confusion_matrix(Y_test, y_hats)
    print(nn_clf.score(X_test, Y_test))
    return cm

def test():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2, random_state=1)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
        
    err,w=fit_NeuralNetwork(X_train,y_train,1e-2,[30, 10],100)
    
    plotErr(err,100)
    
    cM=confMatrix(X_test,y_test,w)
    
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)

test()