from dnpy import losses
import numpy as np


class Initializer:

    def __init__(self, name="Base initializer", seed=None):
        self.name = name
        self.seed = seed

    def apply(self, params):
        pass


class Zeros(Initializer):

    def __init__(self, name="Zeros"):
        super().__init__(name=name, seed=None)

    def apply(self, params, keys=None):
        keys = keys if keys else params.keys()
        for k in keys:
            params[k] = np.zeros(params[k].shape)


class Ones(Initializer):

    def __init__(self, name="Ones", seed=None):
        super().__init__(name=name, seed=seed)

    def apply(self, params, keys=None):
        keys = keys if keys else params.keys()
        for k in keys:
            params[k] = np.ones(params[k].shape)


class RandomNormal(Initializer):

    def __init__(self, name="RandomNormal", mean=0.0, stddev=0.1, seed=None):
        super().__init__(name=name, seed=seed)
        self.mean = mean
        self.stddev = stddev

    def apply(self, params, keys=None):
        keys = keys if keys else params.keys()
        for k in keys:
            params[k] = self.stddev * np.random.randn(*params[k].shape) + self.mean


class RandomUniform(Initializer):

    def __init__(self, name="RandomUniform", minval=-0.05, maxval=0.05, seed=None):
        super().__init__(name=name, seed=seed)
        self.minval = minval
        self.maxval = maxval

    def apply(self, params, keys=None):
        keys = keys if keys else params.keys()
        for k in keys:
            # params[k] = (self.maxval-self.minval) * np.random.rand(*params[k].shape) + self.minval
            params[k] = 2.0 * np.random.random(params[k].shape) - 1.0
