import numpy as np
from dnpy import initializers


class Layer:
    def __init__(self, name):
        self.name = name

        self.input = None
        self.output = None
        self.delta = None

        self.params = {}
        self.grads = {}

        self.parent = None
        self.oshape = None

    def initialize(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def print_stats(self, print_tensors=False):
            print(f"\t=> [DEBUG]: {self.name} layer:")
            if self.parent is not None:
                print(f"\t\t [input]\tshape={self.parent.output.shape}; max={float(np.max(self.parent.output))}; min={float(np.min(self.parent.output))}; avg={float(np.mean(self.parent.output))}")
                if print_tensors: print(self.parent.output)

            print(f"\t\t [output]\tshape={self.output.shape}; max={float(np.max(self.output))}; min={float(np.min(self.output))}; avg={float(np.mean(self.output))}")
            if print_tensors: print(self.output)

            if self.delta is not None:
                print(f"\t\t [delta]\tshape={self.delta.shape}; max={float(np.max(self.delta))}; min={float(np.min(self.delta))}; avg={float(np.mean(self.delta))}")
                if print_tensors: print(self.delta)

            for k in self.params.keys():
                print(f"\t\t [{k}]\tshape={self.params[k].shape}; max={float(np.max(self.params[k]))}; min={float(np.min(self.params[k]))}; avg={float(np.mean(self.params[k]))}")
                if print_tensors: print(self.params[k])

            for k in self.grads.keys():
                print(f"\t\t [{k}]\tshape={self.grads[k].shape}; max={float(np.max(self.grads[k]))}; min={float(np.min(self.grads[k]))}; avg={float(np.mean(self.grads[k]))}")
                if print_tensors: print(self.grads[k])


class Input(Layer):

    def __init__(self, shape):
        super().__init__(name="Input")
        self.oshape = shape

    def forward(self):
        self.output = self.input

    def backward(self):
        pass


class Dense(Layer):

    def __init__(self, l_in, units, kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None):
        super().__init__(name="Dense")
        self.parent = l_in
        self.oshape = (units, 1)
        self.units = units

        # Params and grads
        self.params = {'w1': np.zeros((self.parent.oshape[0], self.units)),
                       'b1': np.zeros((self.units, 1))}
        self.grads = {'g_w1': np.zeros_like(self.params['w1']),
                      'g_b1': np.zeros_like(self.params['b1'])}

        # Initialization: param
        if kernel_initializer is None:
            self.kernel_initializer = initializers.RandomNormal()

        # Initialization: bias
        if bias_initializer is None:
            self.bias_initializer = initializers.Zeros()

        # Add regularizers
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def initialize(self):
        self.kernel_initializer.apply(self.params, ['w1'])
        self.bias_initializer.apply(self.params, ['b1'])

    def forward(self):
        self.output = np.dot(self.params['w1'].T, self.parent.output) + self.params['b1']

    def backward(self):
        # Each layer sets the delta of their parent (13,m)=>(10,m)=>(1,m)=>(1,1)
        self.parent.delta = np.dot(self.params['w1'], self.delta)

        # Compute gradients
        m = self.output.shape[-1]
        g_w1 = np.dot(self.parent.output, self.delta.T)
        g_b1 = np.sum(self.delta, axis=1, keepdims=True)

        # Add regularizers (if needed)
        if self.kernel_regularizer:
            g_w1 += self.kernel_regularizer.backward(self.params['w1'])
        if self.bias_regularizer:
            g_b1 += self.bias_regularizer.backward(self.params['b1'])

        self.grads['g_w1'] += g_w1/m
        self.grads['g_b1'] += g_b1/m


class Relu(Layer):

    def __init__(self, l_in):
        super().__init__(name="Relu")
        self.parent = l_in

        self.oshape = self.parent.oshape
        self.gate = None

    def forward(self):
        self.gate = (self.parent.output > 0).astype(float)
        self.output = self.gate * self.parent.output

    def backward(self):
        # Each layer sets the delta of their parent (13,m)=>(10,m)=>(1,m)=>(1,1)
        self.parent.delta = self.gate * self.delta


class Sigmoid(Layer):

    def __init__(self, l_in):
        super().__init__(name="Sigmoid")
        self.parent = l_in

        self.oshape = self.parent.oshape

    def forward(self):
        self.output = 1.0 / (1.0 + np.exp(-self.parent.output))

    def backward(self):
        # Each layer sets the delta of their parent (13,m)=>(10,m)=>(1,m)=>(1,1)
        self.parent.delta = self.delta * (self.output * (1 - self.output))


class Softmax(Layer):

    def __init__(self, l_in, stable=True):
        super().__init__(name="Softmax")
        self.parent = l_in
        self.oshape = self.parent.oshape

        self.stable = stable

    def forward(self):
        if self.stable:
            z = self.parent.output - np.max(self.parent.output, axis=0, keepdims=True)
        else:
            z = self.parent.output

        exps = np.exp(z)
        sums = np.sum(exps, axis=0, keepdims=True)
        self.output = exps/sums

    def backward(self):
        self.parent.delta = np.zeros_like(self.output)
        m = self.output.shape[-1]
        for i in range(m):
            SM = self.output[:, i].reshape((-1, 1))
            jac = np.diagflat(self.output[:, i]) - np.dot(SM, SM.T)
            self.parent.delta[:, i] = np.dot(jac, self.delta[:, i])

