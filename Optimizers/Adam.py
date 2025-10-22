import numpy as np

from Optimizers.Optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = []
        self.v = []

    def update(self, params, grads):
        if not self.m :
            self.m = [np.zeros_like(p) for layer in params for p in layer]
            self.v = [np.zeros_like(p) for layer in params for p in layer]

        self.t += 1

        flat_params = [p for layer in params for p in layer]
        flat_grads = [g for layer in grads for g in layer]

        for i, (p, grad) in enumerate(zip(flat_params, flat_grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(grad)

            m_reg = self.m[i] / (1 - self.beta1**self.t)
            v_reg = self.v[i] / (1 - self.beta2**self.t)

            flat_params[i] -= self.learning_rate * (m_reg / (np.sqrt(v_reg) + self.epsilon))