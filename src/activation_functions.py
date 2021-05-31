import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def relu(x):
    return np.multiply(x, (x > 0))

def relu_derivative(x):
    return (x > 0) * 1