import copy
import numpy as np
from dnpy.layers import Layer, MLayer
from dnpy import initializers, utils


class MPWLayer(MLayer):

    def __init__(self, l_in, operator, name):
        super().__init__(l_in, name=name)
        self.operator = operator

    def forward(self):
        self.output = np.array(self.parents[0].output)
        for i in range(1, len(self.parents)):
            self.output = self.operator(self.output, self.parents[i].output)

    def backward(self):
        for i in range(len(self.parents)):
            self.parents[i].delta = self.delta


class Add(MPWLayer):

    def __init__(self, l_in, name="Add"):
        super().__init__(l_in, operator=np.add, name=name)


class Subtract(MPWLayer):

    def __init__(self, l_in, name="Subtract"):
        super().__init__(l_in, operator=np.subtract, name=name)


class Multiply(MPWLayer):

    def __init__(self, l_in, name="Multiply"):
        super().__init__(l_in, operator=np.multiply, name=name)


class Divide(MPWLayer):

    def __init__(self, l_in, name="Divide"):
        super().__init__(l_in, operator=np.divide, name=name)


class Power(MPWLayer):

    def __init__(self, l_in, name="Power"):
        super().__init__(l_in, operator=np.power, name=name)


class Maximum(MPWLayer):

    def __init__(self, l_in, name="Maximum"):
        super().__init__(l_in, operator=np.maximum, name=name)


class Minimum(MPWLayer):

    def __init__(self, l_in, name="Minimum"):
        super().__init__(l_in, operator=np.minimum, name=name)


class Average(MLayer):

    def __init__(self, l_in, name="Average"):
        super().__init__(l_in, name=name)

    def forward(self):
        tmp = []
        for i in range(0, len(self.parents)):
            tmp.append(self.parents[i].output)
        self.output = np.sum(tmp, axis=0) / len(self.parents)

    def backward(self):
        for i in range(len(self.parents)):
            self.parents[i].delta = self.delta / len(self.parents)


class Concatenate(MLayer):
    """
    Not tested.
    """
    def __init__(self, l_in, axis=0, name="Concatenate"):
        super().__init__(l_in, name=name)
        self.axis = axis + 1

    def forward(self):
        tmp = []
        for i in range(0, len(self.parents)):
            tmp.append(self.parents[i].output)
        self.output = np.concatenate(tmp, axis=self.axis)

    def backward(self):
        last_idx = 0
        for i in range(len(self.parents)):
            dim = self.parents[i].output.shape[self.axis]
            self.parents[i].delta = np.take(self.delta, slice(last_idx, last_idx+dim), axis=self.axis)
            last_idx += dim
