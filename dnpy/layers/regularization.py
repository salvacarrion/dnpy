import copy
import numpy as np
from dnpy.layers import Layer
from dnpy import initializers, utils


class Dropout(Layer):

    def __init__(self, l_in, rate=0.5, name="Dropout"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.oshape = self.parents[0].oshape
        self.rate = rate
        self.gate = None

    def forward(self):
        if self.training:
            self.gate = (np.random.random(self.parents[0].output.shape) > self.rate).astype(float)
            self.output = self.parents[0].output * self.gate
        else:
            self.output = self.parents[0].output * (1-self.rate)

    def backward(self):
        self.parents[0].delta = self.delta * self.gate


class GaussianNoise(Layer):

    def __init__(self, l_in, mean=0.0, stddev=1.0, name="GaussianNoise"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.mean = mean
        self.stddev = stddev

        self.oshape = self.parents[0].oshape

    def forward(self):
        if self.training:
            noise = np.random.normal(loc=self.mean, scale=self.stddev, size=self.parents[0].output.shape)
            self.output = self.parents[0].output + noise
        else:
            self.output = self.parents[0].output

    def backward(self):
        self.parents[0].delta = np.array(self.delta)


class BatchNorm(Layer):

    def __init__(self, l_in, momentum=0.99, bias_correction=False, gamma_initializer=None,
                 beta_initializer=None, name="BatchNorm"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.oshape = self.parents[0].oshape

        # Params and grads
        self.params = {'gamma': np.ones(self.parents[0].oshape),
                       'beta': np.zeros(self.parents[0].oshape),
                       'moving_mu': np.zeros(self.parents[0].oshape),
                       'moving_var': np.ones(self.parents[0].oshape),
                       }

        self.grads = {'gamma': np.zeros_like(self.params["gamma"]),
                      'beta': np.zeros_like(self.params["beta"])}
        self.cache = {}
        self.fw_steps = 0

        # Constants
        self.momentum = momentum
        self.bias_correction = bias_correction

        # Initialization: gamma
        if gamma_initializer is None:
            self.gamma_initializer = initializers.Ones()
        else:
            self.gamma_initializer = gamma_initializer

        # Initialization: beta
        if beta_initializer is None:
            self.beta_initializer = initializers.Zeros()
        else:
            self.beta_initializer = beta_initializer

    def initialize(self, optimizer=None):
        super().initialize(optimizer=optimizer)

        # Initialize params
        self.gamma_initializer.apply(self.params, ['gamma'])
        self.beta_initializer.apply(self.params, ['beta'])

    def forward(self):
        x = self.parents[0].output

        if self.training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            # Get moving average/variance
            self.fw_steps += 1
            # Add the bias_correction part to use the implicit correction
            if self.bias_correction and self.fw_steps == 1:
                moving_mu = mu
                moving_var = var
            else:
                # Compute exponentially weighted average (aka moving average)
                # No bias correction => Use the implicit "correction" of starting with mu=zero and var=one
                # Bias correction => Simply apply weighted average
                moving_mu = self.momentum * self.params['moving_mu'] + (1.0 - self.momentum) * mu
                moving_var = self.momentum * self.params['moving_var'] + (1.0 - self.momentum) * var

            # Compute bias correction
            # (Not working! It's too aggressive)
            if self.bias_correction and self.fw_steps <= 1000:  # Limit set to prevent overflow
                bias_correction = 1.0/(1-self.momentum**self.fw_steps)
                moving_mu *= bias_correction
                moving_var *= bias_correction

            # Save moving averages
            self.params['moving_mu'] = moving_mu
            self.params['moving_var'] = moving_var
        else:
            mu = self.params['moving_mu']
            var = self.params['moving_var']

        inv_var = np.sqrt(var + self.epsilon)
        x_norm = (x-mu)/inv_var

        self.output = self.params["gamma"] * x_norm + self.params["beta"]

        # Cache vars
        self.cache['mu'] = mu
        self.cache['var'] = var
        self.cache['inv_var'] = inv_var
        self.cache['x_norm'] = x_norm

    def backward(self):
        m = self.output.shape[0]
        mu, var = self.cache['mu'], self.cache['var']
        inv_var, x_norm = self.cache['inv_var'], self.cache['x_norm']

        dgamma = self.delta * mu
        dbeta = self.delta  # * 1.0
        dxnorm = self.delta * self.params["gamma"]

        df_xi = (1.0/m) * inv_var * (
                (m * dxnorm)
                - (np.sum(dxnorm, axis=0, keepdims=True))
                - (x_norm * np.sum(dxnorm*x_norm, axis=0, keepdims=True))
        )

        self.parents[0].delta = df_xi
        self.grads["gamma"] += np.sum(dgamma, axis=0)
        self.grads["beta"] += np.sum(dbeta, axis=0)
