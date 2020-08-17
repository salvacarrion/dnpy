import copy
import numpy as np
from dnpy.layers import Layer
from dnpy import initializers, utils


class LeakyRelu(Layer):

    def __init__(self, l_in, alpha=0.3, name="LeakyRelu"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.alpha = alpha
        self.oshape = self.parents[0].oshape
        self.gate = None

    def forward(self):
        self.gate = (self.parents[0].output > 0)
        self.output = np.where(self.gate, self.parents[0].output, self.parents[0].output * self.alpha)

    def backward(self):
        self.parents[0].delta = np.where(self.gate, self.delta, self.delta * self.alpha)


class Relu(LeakyRelu):

    def __init__(self, l_in, name="Relu"):
        super().__init__(l_in, alpha=0.0, name=name)


class PRelu(Layer):

    def __init__(self, l_in, alpha_initializer=None, alpha_regularizer=None, name="LeakyRelu"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.oshape = self.parents[0].oshape
        self.gate = None

        # Params and grads
        self.params = {'alpha': np.zeros(self.oshape)}
        self.grads = {'alpha': np.zeros_like(self.params['alpha'])}

        # Initialization: bias
        if alpha_initializer is None:
            self.alpha_initializer = initializers.Zeros()
        else:
            self.alpha_initializer = alpha_initializer

        # Add regularizers
        self.alpha_regularizer = alpha_regularizer

    def initialize(self, optimizer=None):
        super().initialize(optimizer=optimizer)

        # Initialize params
        self.alpha_initializer.apply(self.params, ['alpha'])

    def forward(self):
        self.gate = (self.parents[0].output > 0)
        self.output = np.where(self.gate, self.parents[0].output, self.parents[0].output * self.params['alpha'])

    def backward(self):
        self.parents[0].delta = np.where(self.gate, self.delta, self.delta * self.params['alpha'])
        # Select deltas with negative date (inv&mult)
        self.grads['alpha'] += np.mean(self.delta*np.invert(self.gate).astype(float), axis=0)


class Sigmoid(Layer):

    def __init__(self, l_in, name="Sigmoid"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.oshape = self.parents[0].oshape

    def forward(self):
        self.output = 1.0 / (1.0 + np.exp(-self.parents[0].output))

    def backward(self):
        self.parents[0].delta = self.delta * (self.output * (1 - self.output))


class Tanh(Layer):

    def __init__(self, l_in, name="Tanh"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.oshape = self.parents[0].oshape

    def forward(self):
        a = np.exp(self.parents[0].output)
        b = np.exp(-self.parents[0].output)
        self.output = (a - b) / (a + b)

    def backward(self):
        self.parents[0].delta = self.delta * (1 - self.output**2)


class Softmax(Layer):

    def __init__(self, l_in, stable=True, name="Softmax"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Check layer compatibility
        if len(l_in.oshape) != 1:
            raise ValueError(f"Expected a 1D layer ({self.name})")

        self.oshape = self.parents[0].oshape

        self.stable = stable
        self.ce_loss = False

    def forward(self):
        if self.stable:
            z = self.parents[0].output - np.max(self.parents[0].output, axis=1, keepdims=True)
        else:
            z = self.parents[0].output

        exps = np.exp(z)
        sums = np.sum(exps, axis=1, keepdims=True)
        self.output = exps/sums

    def backward(self):
        if self.ce_loss:  # Only valid for a Cross-Entropy loss
            self.parents[0].delta = self.delta
        else:  # Generic
            self.parents[0].delta = np.zeros_like(self.output)
            m = self.output.shape[0]
            for i in range(m):
                SM = self.output[i, :].reshape((-1, 1))
                jac = np.diagflat(self.output[i, :]) - np.dot(SM, SM.T)
                self.parents[0].delta[i, :] = np.dot(self.delta[i, :], jac)


class LogSoftmax(Layer):
    # softmax + CE = log(softmax) + CE(without log) = LogSoftmax + NNL
    def __init__(self, l_in, stable=True, name="LogSoftmax"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Check layer compatibility
        if len(l_in.oshape) != 1:
            raise ValueError(f"Expected a 1D layer ({self.name})")

        self.oshape = self.parents[0].oshape

        self.stable = stable

    def forward(self):
        if self.stable:
            z = self.parents[0].output - np.max(self.parents[0].output, axis=1, keepdims=True)
        else:
            z = self.parents[0].output

        exps = np.exp(z)
        sums = np.sum(exps, axis=1, keepdims=True)
        self.output = z - np.log(sums+self.epsilon)  # np.exp(output) == softmax

    def backward(self):
        self.parents[0].delta = self.delta  # exp(y_pred) - y_target
