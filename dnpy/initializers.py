from dnpy import losses
import numpy as np


class Initializer:

    def __init__(self, name="Base initializer", seed=None):
        self.name = name
        self.seed = seed

    def apply(self, params):
        pass


class Constant(Initializer):

    def __init__(self, fill_value=1.0, name="Constant"):
        super().__init__(name=name, seed=None)
        self.fill_value = fill_value

    def apply(self, params, keys=None):
        keys = keys if keys else params.keys()
        for k in keys:
            params[k] = np.full(params[k].shape, fill_value=self.fill_value)


class Zeros(Constant):

    def __init__(self, name="Zeros"):
        super().__init__(fill_value=0.0, name=name)


class Ones(Constant):

    def __init__(self, name="Ones"):
        super().__init__(fill_value=1.0, name=name)


class RandomNormal(Initializer):

    def __init__(self, mean=0.0, stddev=0.05, name="RandomNormal", seed=None):
        super().__init__(name=name, seed=seed)
        self.mean = mean
        self.stddev = stddev

    def apply(self, params, keys=None):
        keys = keys if keys else params.keys()
        for k in keys:
            params[k] = self.stddev * np.random.randn(*params[k].shape) + self.mean


class RandomUniform(Initializer):

    def __init__(self, minval=-0.05, maxval=0.05, name="RandomUniform", seed=None):
        super().__init__(name=name, seed=seed)
        self.minval = minval
        self.maxval = maxval

    def apply(self, params, keys=None):
        keys = keys if keys else params.keys()
        for k in keys:
            params[k] = (self.maxval-self.minval) * np.random.random(params[k].shape) + self.minval


class GlorotNormal(Initializer):

    def __init__(self, fan_in=None, fan_out=None, name="GlorotNormal", seed=None):
        super().__init__(name=name, seed=seed)
        self.fan_in = fan_in
        self.fan_out = fan_out

    def apply(self, params, keys=None):
        var = np.sqrt(2.0 / (self.fan_in + self.fan_out))
        keys = keys if keys else params.keys()
        for k in keys:
            params[k] = np.random.randn(*params[k].shape)
            params[k] *= var


class GlorotUniform(Initializer):

    def __init__(self, fan_in=None, fan_out=None, minval=-0.05, maxval=0.05, name="GlorotUniform", seed=None):
        super().__init__(name=name, seed=seed)
        self.minval = minval
        self.maxval = maxval
        self.fan_in = fan_in
        self.fan_out = fan_out

    def apply(self, params, keys=None):
        var = np.sqrt(2.0 / (self.fan_in + self.fan_out))
        keys = keys if keys else params.keys()
        for k in keys:
            params[k] = (self.maxval-self.minval) * np.random.random(params[k].shape) + self.minval
            params[k] *= var


class HeNormal(Initializer):

    def __init__(self, fan_in=None, fan_out=None, name="HeNormal", seed=None):
        super().__init__(name=name, seed=seed)
        self.fan_in = fan_in
        self.fan_out = fan_out

    def apply(self, params, keys=None):
        var = np.sqrt(2.0 / self.fan_in)
        keys = keys if keys else params.keys()
        for k in keys:
            params[k] = np.random.randn(*params[k].shape)
            params[k] *= var
