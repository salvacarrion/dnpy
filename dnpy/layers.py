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

    def print_stats(self):
        if self.parent is None:
            print(f"\t=> [DEBUG]{self.name}: Nothing to print")
        else:
            print(f"\t=> [DEBUG]: {self.name} layer:")
            print(f"\t\t [input]\tshape={self.parent.output.shape}; max={float(np.max(self.parent.output))}; min={float(np.min(self.parent.output))}; avg={float(np.mean(self.parent.output))}")
            print(f"\t\t [output]\tshape={self.output.shape}; max={float(np.max(self.output))}; min={float(np.min(self.output))}; avg={float(np.mean(self.output))}")
            print(f"\t\t [delta]\tshape={self.delta.shape}; max={float(np.max(self.delta))}; min={float(np.min(self.delta))}; avg={float(np.mean(self.delta))}")

            for k in self.params.keys():
                print(f"\t\t [{k}]\tshape={self.params[k].shape}; max={float(np.max(self.params[k]))}; min={float(np.min(self.params[k]))}; avg={float(np.mean(self.params[k]))}")

            for k in self.grads.keys():
                print(f"\t\t [{k}]\tshape={self.grads[k].shape}; max={float(np.max(self.grads[k]))}; min={float(np.min(self.grads[k]))}; avg={float(np.mean(self.grads[k]))}")


class Input(Layer):

    def __init__(self, shape):
        super().__init__(name="Input")
        self.oshape = shape

    def forward(self):
        self.output = self.input

    def backward(self):
        pass


class Dense(Layer):

    def __init__(self, l_in, units, kernel_initializer=None, bias_initializer=None):
        super().__init__(name="Dense")
        self.parent = l_in
        self.oshape = (units, 1)
        self.units = units

        # Params and grads
        self.params = {'w1': np.zeros((self.units, self.parent.oshape[0])),
                       'b1': np.zeros((self.units, 1))}
        self.grads = {'g_w1': np.zeros_like(self.params['w1']),
                      'g_b1': np.zeros_like(self.params['b1'])}

        # Initialization: param
        if kernel_initializer is None:
            self.kernel_initializer = initializers.RandomUniform()

        # Initialization: bias
        if bias_initializer is None:
            self.bias_initializer = initializers.Ones()

    def initialize(self):
        self.kernel_initializer.apply(self.params, ['w1'])
        self.bias_initializer.apply(self.params, ['b1'])

    def forward(self):
        self.output = np.dot(self.params['w1'], self.parent.output) + self.params['b1']

    def backward(self):
        # Each layer sets the delta of their parent (13,m)=>(10,m)=>(1,m)=>(1,1)
        self.parent.delta = np.dot(self.params['w1'].T, self.delta)

        # Compute gradients
        m = self.output.shape[-1]
        # self.grads['g_w1'] += np.dot(self.delta, self.parent.output.T)/m
        self.grads['g_b1'] += np.sum(self.delta, axis=1, keepdims=True)/m


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
        self.tmp = None

    def forward(self):
        self.output = 1.0 / (1.0 + np.exp(-self.parent.output))
        self.tmp = self.output

    def backward(self):
        # Each layer sets the delta of their parent (13,m)=>(10,m)=>(1,m)=>(1,1)
        self.parent.delta = self.delta * (self.tmp * (1 - self.tmp))


class Softmax(Layer):

    def __init__(self, l_in):
        super().__init__(name="Softmax")
        self.parent = l_in

        self.oshape = self.parent.oshape

    def forward(self):
        exps = np.exp(self.parent.output)
        sums = np.sum(np.exp(self.parent.output), axis=0, keepdims=True)
        self.output = exps/sums

    def backward(self):
        m = self.output.shape[-1]
        tmp = []
        for i in range(m):
            SM = self.output[:, i].reshape((-1, 1))
            jaccobian = np.diagflat(self.parent.output[:, i]) - np.dot(SM, SM.T)
            delta_i = np.dot(jaccobian, self.delta[:, i])
            tmp.append(delta_i)
        self.parent.delta = np.array(tmp).T
