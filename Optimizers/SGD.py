from Optimizers.Optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        coef, bias = params
        grad_coef, grad_bias = grads

        for i in range(len(coef)):
            coef[i] -= self.learning_rate * grad_coef[i]
            bias[i] -= self.learning_rate * grad_bias[i]