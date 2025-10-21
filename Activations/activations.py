import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def linear(x):
    return x

def derivative_relu(x):
    return (x > 0).astype(float)

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def derivative_tanh(x):
    return 1 - tanh(x)**2

def derivative_linear(x):
    return np.ones_like(x)