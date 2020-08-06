import copy
import numpy as np
from dnpy.layers import Layer
from dnpy import initializers, utils


class Relu(Layer):

    def __init__(self, l_in, name="Relu"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.oshape = self.parents[0].oshape
        self.gate = None

    def forward(self):
        self.gate = (self.parents[0].output > 0).astype(float)
        self.output = self.gate * self.parents[0].output

    def backward(self):
        # Each layer sets the delta of their parent (m,13)=>(m, 10)=>(m, 1)=>(1,1)
        self.parents[0].delta = self.gate * self.delta


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
                self.parents[0].delta[i, :] = np.dot(jac, self.delta[i, :])

