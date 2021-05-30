import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def relu(x):
        x[x <= 0] = 0
        return x 

def relu_derivative(x):
    x[x<=0] = 0
    x[x > 0] = 1
    return x