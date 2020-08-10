import numpy as np


class Optimizer:
    def __init__(self, name="Base optimizer"):
        self.name = name

    def initialize(self, params):
        pass

    def apply(self, params, grads, step_i=None):
        pass


class SGD(Optimizer):

    def __init__(self, lr=0.001, momentum=0.0, nesterov=False, bias_correction=False):
        super().__init__(name="Momentum")
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.bias_correction = bias_correction

        # Params
        self.V = {}

    def initialize(self, params):
        for k in params.keys():
            self.V[k] = np.zeros_like(params[k])

    def apply(self, params, grads, step_i=None):
        for k in params.keys():
            # Exclude no trainable params
            if k not in grads:
                continue

            # Momentum
            v_prev = self.V[k]
            self.V[k] = self.momentum * self.V[k] + (1 - self.momentum) * grads[k]

            # Bias correction. It is not common to use it here
            # Note: Vanilla SGD does not need bias correction
            if self.bias_correction and self.momentum != 0.0:
                b_correction = 1.0/(1.0 - self.lr**step_i)
                self.V[k] *= b_correction

            # Compute update
            if not self.nesterov:
                new_v = self.V[k]
            else:
                # Not sure if this is correct
                new_v = -self.momentum * v_prev + (1 + self.momentum) * self.V[k]

            # Step
            params[k] -= self.lr * new_v


class RMSProp(Optimizer):

    def __init__(self, lr=0.001, rho=0.9, epsilon=10e-8):
        super().__init__(name="RMSProp")
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

        # Params
        self.S = {}

    def initialize(self, params):
        for k in params.keys():
            self.S[k] = np.zeros_like(params[k])

    def apply(self, params, grads, step_i=None):
        for k in params.keys():
            # Exclude no trainable params
            if k not in grads:
                continue

            # Momentum
            self.S[k] = self.rho * self.S[k] + (1.0 - self.rho) * grads[k]**2

            # Step
            new_v = grads[k]/(np.sqrt(self.S[k]+self.epsilon))
            params[k] -= self.lr * new_v


class Adam(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, bias_correction=False):
        super().__init__(name="Adam")
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.bias_correction = bias_correction

        # Params
        self.V = {}
        self.S = {}

    def initialize(self, params):
        for k in params.keys():
            self.V[k] = np.zeros_like(params[k])
            self.S[k] = np.zeros_like(params[k])

    def apply(self, params, grads, step_i=None):
        for k in params.keys():
            # Exclude no trainable params
            if k not in grads:
                continue

            # Momentum
            self.V[k] = self.beta_1 * self.V[k] + (1.0 - self.beta_1) * grads[k]
            self.S[k] = self.beta_2 * self.S[k] + (1.0 - self.beta_2) * grads[k] ** 2

            # Bias correction
            if self.bias_correction:
                b_correction = 1.0 / (1.0 - self.lr ** step_i)
                vk_corrected = self.V[k] * b_correction
                sk_corrected = self.S[k] * b_correction
            else:
                vk_corrected, sk_corrected = self.V[k], self.S[k]

            # Step
            new_v = vk_corrected / (np.sqrt(sk_corrected + self.epsilon))
            params[k] -= self.lr * new_v
