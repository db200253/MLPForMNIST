import numpy as np
from Optimizers.Optimizer import Optimizer
from Activations.activations import *

class MLP:
    def __init__(self, layers, activations, learning_rate, optimizer):
        if not isinstance(layers, list):
            raise TypeError('layers must be a list')
        for layer in layers:
            if not isinstance(layer, int):
                raise TypeError('layers must be a list of integers')
        if not isinstance(optimizer, Optimizer):
            raise TypeError('optimizer must be an Optimizer')
        if len(layers) != len(activations):
            raise ValueError('layers and activations must have the same length')

        self.layers = layers
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.coef = []
        self.bias = []

        for i in range(len(layers) - 1):
            n_in = layers[i]
            n_out = layers[i + 1]
            if activations[i] in ["sigmoid", "tanh"]:
                mat_i = np.random.randn(n_in, n_out) * np.sqrt(1. / n_in)
            elif activations[i] == "relu":
                mat_i = np.random.randn(n_in, n_out) * np.sqrt(2. / n_in)
            else:
                mat_i = np.random.randn(n_in, n_out) * 0.01
            self.coef.append(mat_i)
            vect_i = np.zeros(n_out)
            self.bias.append(vect_i)

        self.activation_funcs = {
            "relu": (relu, derivative_relu),
            "sigmoid": (sigmoid, derivative_sigmoid),
            "softmax": (softmax, None),
            "tanh": (tanh, derivative_tanh),
            "linear": (linear, derivative_linear)
        }

        self.activations = []
        for a in activations:
            self.activations.append(self.activation_funcs[a][0])
        self.back = []
        for a in self.activations:
            self.back.append(self.activation_funcs[a][1])

    def forward(self, x):
        a = x
        for i in range(len(self.layers) - 1):
            zi = np.dot(a, self.coef[i]) + self.bias[i]
            a = self.activations[i](zi)
        z = np.dot(a, self.coef[-1]) + self.bias[-1]
        output = self.activations[-1](z)
        return output