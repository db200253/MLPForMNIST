import numpy as np
from Optimizers.Optimizer import Optimizer
from Activations.activations import *

def loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

class MLP:
    def __init__(self, epochs, tolerance, layers, activations, learning_rate, optimizer):
        if not isinstance(layers, list):
            raise TypeError('layers must be a list')
        for layer in layers:
            if not isinstance(layer, int):
                raise TypeError('layers must be a list of integers')
        if not isinstance(optimizer, Optimizer):
            raise TypeError('optimizer must be an Optimizer')
        if len(layers) != len(activations):
            raise ValueError('layers and activations must have the same length')

        self.epochs = epochs
        self.tolerance = tolerance
        self.layers = layers
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.coef = []
        self.bias = []
        self.grad_coef = []
        self.grad_bias = []

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

        self.activations = [self.activation_funcs[a][0] for a in activations]
        self.back = [self.activation_funcs[a][1] for a in activations]

    def forward(self, x):
        a = x
        for i in range(len(self.layers) - 1):
            zi = np.dot(a, self.coef[i]) + self.bias[i]
            a = self.activations[i](zi)
        z = np.dot(a, self.coef[-1]) + self.bias[-1]
        output = self.activations[-1](z)
        return output

    def backward(self, y_true, y_pred):
        dz = y_pred - y_true
        for i in range(len(self.layers) - 2, -1, -1):
            zi = np.dot(dz, self.coef[i].T)
            dz = dz * self.back[i](zi)
            gc = np.dot(self.activations[i].T, dz)
            gb = np.sum(dz, axis=0)
            self.grad_coef[i] = gc
            self.grad_bias[i] = gb

    def fit(self, x, y, batch_size):
        cost = float('inf')
        permutation = np.random.permutation(len(x))
        x, y = x[permutation], y[permutation]
        for epoch in range(self.epochs):
            prev_cost = cost
            for start in range(0, len(x), batch_size):
                end = min(start + batch_size, len(x))
                x_batch = x[start:end]
                y_batch = y[start:end]
                y_pred = self.forward(x_batch)
                cost = loss(y_pred, y_batch)
                self.backward(y_batch, y_pred)
                self.optimizer.update([self.coef, self.bias], [self.grad_coef, self.grad_bias])
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {cost}")
            if prev_cost - cost < self.tolerance:
                break

    def predict(self, x):
        return self.forward(x)