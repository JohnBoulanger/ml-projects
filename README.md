# Machine Learning Models from Scratch

This repository contains from-scratch implementations of core machine
learning models using only NumPy for the learning algorithms. High-level
libraries such as scikit-learn are used only for dataset loading,
evaluation, and benchmarking.

The goal of this repository is educational: to demonstrate how common
models work internally, rather than to provide production-ready
implementations.

---

## Contents

- `LinearRegression.py`  
  Closed-form linear regression using the normal equation.

- `NeuralNetwork.py`  
  Fully connected feedforward neural network trained with
  backpropagation and stochastic gradient descent.

---

## Linear Regression

### Overview
The linear regression model is implemented using the normal equation:

\[
w = (X^\dagger) y
\]

where \(X^\dagger\) is the Moore–Penrose pseudoinverse of the design
matrix with an explicitly added bias term.

### Key Characteristics
- No gradient descent
- Explicit bias handling
- Mean squared error evaluation
- Suitable for understanding the mathematics of linear regression

---

## Neural Network

### Overview
The neural network is a configurable multilayer perceptron implemented
entirely from scratch. The network supports an arbitrary number of
hidden layers and is trained using backpropagation.

### Key Characteristics
- Manual forward and backward passes
- Bias handled via augmented weight matrices
- Stochastic gradient descent updates
- Error tracking across epochs
- Comparison with `sklearn.neural_network.MLPClassifier`

---

## Dependencies

- Python 3.x
- NumPy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:

```bash
pip install numpy matplotlib seaborn scikit-learn
