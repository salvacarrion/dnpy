import copy
import numpy as np
from dnpy.layers import Layer
from dnpy import initializers, utils


class Add(Layer):

    def __init__(self, l_in, name="Add"):
        super().__init__(name=name)

        # Check inputs
        if not isinstance(l_in, list):
            raise ValueError("A list of layers is expected")

        # Check number of inputs
        if len(l_in) < 2:
            raise ValueError("A minimum of two inputs is expected")

        # Check if all layers have the same dimension
        dim1 = l_in[0].oshape
        for i in range(1, len(l_in)):
            dim2 = l_in[i].oshape
            if dim1 != dim2:
                raise ValueError(f"Layers with different dimensions: {dim1} vs {dim2}")

        # Add layers
        self.parents = l_in
        self.oshape = self.parents[0].oshape

    def forward(self):
        self.output = np.array(self.parents[0].output)
        for i in range(1, len(self.parents)):
            self.output += self.parents[i].output

    def backward(self):
        for i in range(len(self.parents)):
            self.parents[i].delta = self.delta
