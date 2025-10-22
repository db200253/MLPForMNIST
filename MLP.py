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
            self.grad_coef.append(np.zeros_like(mat_i))
            self.grad_bias.append(np.zeros_like(vect_i))

        self.activation_funcs = {
            "relu": (relu, derivative_relu),
            "sigmoid": (sigmoid, derivative_sigmoid),
            "softmax": (softmax, None),
            "tanh": (tanh, derivative_tanh),
            "linear": (linear, derivative_linear)
        }

        self.activations = [self.activation_funcs[a][0] for a in activations]
        self.back = [self.activation_funcs[a][1] for a in activations]

        self.a_values = []
        self.z_values = []

    def forward(self, x):
        self.a_values = [x]
        self.z_values = []

        a = x
        for i in range(len(self.coef)):
            z = np.dot(a, self.coef[i]) + self.bias[i]
            self.z_values.append(z)
            a = self.activations[i](z)
            self.a_values.append(a)
        return a

    def backward(self, y_true, y_pred):
        m = y_true.shape[0]
        dz = y_pred - y_true
        for i in reversed(range(len(self.coef))):
            a_prev = self.a_values[i]
            z = self.z_values[i]

            if self.back[i] is not None:
                dz *= self.back[i](z)

            self.grad_coef[i] = np.dot(a_prev.T, dz) / m
            self.grad_bias[i] = np.sum(dz, axis=0) / m

            if i > 0:
                dz = np.dot(dz, self.coef[i].T)

    def fit(self, x, y, batch_size):
        cost = float('inf')
        for epoch in range(self.epochs):
            prev_cost = cost
            permutation = np.random.permutation(len(x))
            x, y = x[permutation], y[permutation]

            epoch_preds = []
            epoch_targets = []
            epoch_costs = []

            for start in range(0, len(x), batch_size):
                end = min(start + batch_size, len(x))
                x_batch = x[start:end]
                y_batch = y[start:end]
                y_pred = self.forward(x_batch)
                cost = loss(y_pred, y_batch)
                self.backward(y_batch, y_pred)
                self.optimizer.update([self.coef, self.bias], [self.grad_coef, self.grad_bias])

                epoch_preds.append(np.argmax(y_pred, axis=1))
                epoch_targets.append(np.argmax(y_batch, axis=1))
                epoch_costs.append(cost)
            all_preds = np.concatenate(epoch_preds)
            all_targets = np.concatenate(epoch_targets)
            accuracy = np.mean(all_preds == all_targets)
            cost = np.mean(epoch_costs)

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {cost:.4f}, Accuracy: {accuracy:.2%}")

            if prev_cost - cost < self.tolerance:
                break

    def predict(self, x):
        return self.forward(x)