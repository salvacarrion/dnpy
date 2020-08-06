import copy
import numpy as np
from dnpy import initializers, utils


class Layer:

    def __init__(self, name):
        self.name = name
        self.training = False

        self.parents = []
        self.output = None
        self.delta = None

        self.oshape = None  # To remove

        self.params = {}
        self.grads = {}

        self.epsilon = 10e-8
        self.frozen = False
        self.optimizer = None

        self.index = 0  # For topological sort

    def __str__(self):
        return self.name

    def initialize(self, optimizer=None):
        # Each optimizer must be independent (internal params per layer)
        if optimizer:
            self.optimizer = copy.deepcopy(optimizer)
            self.optimizer.initialize(self.params)

    def forward(self):
        pass

    def backward(self):
        pass

    def print_stats(self, print_tensors=False):
            print(f"\t=> [DEBUG]: {self.name} layer:")
            if self.parents[0] is not None:
                print(f"\t\t [input]\tshape={self.parents[0].output.shape}; max={float(np.max(self.parents[0].output))}; min={float(np.min(self.parents[0].output))}; avg={float(np.mean(self.parents[0].output))}")
                if print_tensors: print(self.parents[0].output)

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


class MLayer(Layer):
    def __init__(self, l_in, name):
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

