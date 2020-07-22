class Optimizer:
    def __init__(self, name="Base optimizer"):
        self.name = name

    def apply(self, params, grads):
        pass


class SGD(Optimizer):

    def __init__(self, lr=0.001):
        super().__init__(name="SGD")
        self.lr = lr

    def apply(self, params, grads):
        for k in params.keys():
            params[k] -= self.lr * grads["g_"+k]


class Momentum(Optimizer):

    def __init__(self, lr=0.001, momentum=0.9):
        super().__init__(name="SGD")
        self.lr = lr
        self.momentum = momentum

    def apply(self, params, grads):
        for k in params.keys():
            v = self.momentum * grads["g_" + k] + (1 - self.momentum) * grads["g_" + k]
            params[k] -= self.lr * v
