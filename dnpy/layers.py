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


class Input(Layer):

    def __init__(self, shape, name="Input"):
        super().__init__(name=name)
        self.oshape = shape

    def forward(self):
        pass

    def backward(self):
        pass


class Dense(Layer):

    def __init__(self, l_in, units, kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None, name="Dense"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Check layer compatibility
        if len(l_in.oshape) != 1:
            raise ValueError(f"Expected a 1D layer ({self.name})")

        # Tricky
        if units == -1:
            units = l_in.oshape[0]

        self.oshape = (units,)
        self.units = units

        # Params and grads
        self.params = {'w1': np.zeros((self.parents[0].oshape[0], self.units)),
                       'b1': np.zeros((1, self.units))}
        self.grads = {'w1': np.zeros_like(self.params['w1']),
                      'b1': np.zeros_like(self.params['b1'])}

        # Initialization: param
        if kernel_initializer is None:
            fan_in, fan_out = self.params['w1'].shape
            self.kernel_initializer = initializers.HeNormal(fan_in=fan_in, fan_out=fan_out)
        else:
            self.kernel_initializer = kernel_initializer

        # Initialization: bias
        if bias_initializer is None:
            self.bias_initializer = initializers.Zeros()
        else:
            self.bias_initializer = kernel_initializer

        # Add regularizers
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def initialize(self, optimizer=None):
        super().initialize(optimizer=optimizer)

        # Initialize params
        self.kernel_initializer.apply(self.params, ['w1'])
        self.bias_initializer.apply(self.params, ['b1'])

    def forward(self):
        self.output = np.dot(self.parents[0].output, self.params['w1']) + self.params['b1']

    def backward(self):
        # Each layer sets the delta of their parent (m,13)=>(m, 10)=>(m, 1)=>(1,1)
        self.parents[0].delta = np.dot(self.delta, self.params['w1'].T)

        # Compute gradients
        m = self.output.shape[0]
        g_w1 = np.dot(self.parents[0].output.T, self.delta)
        g_b1 = np.sum(self.delta, axis=0, keepdims=True)

        # Add regularizers (if needed)
        if self.kernel_regularizer:
            g_w1 += self.kernel_regularizer.backward(self.params['w1'])
        if self.bias_regularizer:
            g_b1 += self.bias_regularizer.backward(self.params['b1'])

        self.grads['w1'] += g_w1/m
        self.grads['b1'] += g_b1/m


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
            self.output = self.parents[0].output

    def backward(self):
        self.parents[0].delta = self.delta * self.gate


class BatchNorm(Layer):

    def __init__(self, l_in, momentum=0.99, bias_correction=False, gamma_initializer=None,
                 beta_initializer=None, name="BatchNorm"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.oshape = self.parents[0].oshape

        # Params and grads
        self.params = {'gamma': np.ones(self.parents[0].oshape),
                       'beta': np.zeros(self.parents[0].oshape)}
        self.params_fixed = {'moving_mu': np.zeros(self.parents[0].oshape),
                             'moving_var': np.ones(self.parents[0].oshape)}
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
                moving_mu = self.momentum * self.params_fixed['moving_mu'] + (1.0 - self.momentum) * mu
                moving_var = self.momentum * self.params_fixed['moving_var'] + (1.0 - self.momentum) * var

            # Compute bias correction
            # (Not working! It's too aggressive)
            if self.bias_correction and self.fw_steps <= 1000:  # Limit set to prevent overflow
                bias_correction = 1.0/(1-self.momentum**self.fw_steps)
                moving_mu *= bias_correction
                moving_var *= bias_correction

            # Save moving averages
            self.params_fixed['moving_mu'] = moving_mu
            self.params_fixed['moving_var'] = moving_var
        else:
            mu = self.params_fixed['moving_mu']
            var = self.params_fixed['moving_var']

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


class Reshape(Layer):

    def __init__(self, l_in, shape, name="Reshape"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Check if shape is inferred
        if shape == -1 or shape[0] == -1:
            shape = (int(np.prod(self.parents[0].oshape)),)

        # Check layer compatibility
        if np.prod(self.parents[0].oshape) != np.prod(shape):
            raise ValueError(f"Not compatible shapes ({self.name})")

        self.oshape = shape

    def forward(self):
        new_shape = (-1, *self.oshape)  # Due to batch
        self.output = np.reshape(self.parents[0].output, newshape=new_shape)

    def backward(self):
        new_shape = (-1, *self.parents[0].oshape)  # Due to batch
        self.parents[0].delta = np.reshape(self.delta, newshape=new_shape)


class Flatten(Reshape):

    def __init__(self, l_in, name="Flatten"):
        super().__init__(l_in, shape=(-1,), name=name)


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


class Conv2D(Layer):

    def __init__(self, l_in, filters, kernel_size, strides=(1, 1), padding="same",
                 dilation_rate=(1, 1),
                 kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None, name="Conv2D"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Check layer compatibility
        if len(l_in.oshape) != 3:
            raise ValueError(f"Expected a 3D layer ({self.name})")

        # Params
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate

        # Compute sizes
        # Note: First compute the output size, then the padding (ignore the typical formula)
        _oshape = utils.get_output(input_size=l_in.oshape[1:], kernel_size=kernel_size,
                                   strides=strides, padding=padding, dilation_rate=dilation_rate)
        _pads = utils.get_padding(padding=padding, input_size=l_in.oshape[1:], output_size=_oshape[1:],
                                  kernel_size=kernel_size, strides=strides)
        # Output / Pads size
        self.pads = tuple(_pads)  # height, width
        self.oshape = tuple([self.filters] + _oshape.tolist())

        # Specific pads
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = utils.get_padding_tblr(self.pads)

        # Params and grads
        channels = l_in.oshape[0]
        self.params = {'w1': np.zeros((self.filters, channels, *self.kernel_size)),
                       'b1': np.zeros((self.filters,))}
        self.grads = {'w1': np.zeros_like(self.params['w1']),
                      'b1': np.zeros_like(self.params['b1'])}

        # Initialization: param
        if kernel_initializer is None:
            # There are "num input feature maps * filter height * filter width" inputs to each hidden unit
            out_fm, in_fm, f_height, f_width = self.params['w1'].shape
            fan_in = in_fm*f_height*f_width

            #  Each unit in the lower layer receives a gradient from:
            #  "num output feature maps * filter height * filter width" / pooling size
            fan_out = out_fm*f_height*f_width
            self.kernel_initializer = initializers.HeNormal(fan_in=fan_in, fan_out=fan_out)
        else:
            self.kernel_initializer = kernel_initializer

        # Initialization: bias
        if bias_initializer is None:
            self.bias_initializer = initializers.Zeros()
        else:
            self.bias_initializer = kernel_initializer

        # Add regularizers
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Cache
        self.shape_pad_in = None
        self.in_fmap = None

    def initialize(self, optimizer=None):
        super().initialize(optimizer=optimizer)

        # This is not really correct since "w1" contains many filters
        # Initialize params
        self.kernel_initializer.apply(self.params, ['w1'])
        self.bias_initializer.apply(self.params, ['b1'])

    def forward(self):
        m = self.parents[0].output.shape[0]  # Batches
        self.output = np.zeros((m, *self.oshape))

        # Reshape input adding paddings
        self.shape_pad_in = tuple(np.array(self.parents[0].oshape) + np.array([0, *self.pads]))
        self.in_fmap = np.zeros((m, *self.shape_pad_in))
        self.in_fmap[:, :, self.pad_top:(self.shape_pad_in[1]-self.pad_bottom), self.pad_left:(self.shape_pad_in[2]-self.pad_right)] = self.parents[0].output

        for fi in range(self.oshape[0]):  # For each filter
            for y in range(self.oshape[1]):  # Walk output's height
                in_y = y * self.strides[0]
                for x in range(self.oshape[2]):  # Walk output's width
                    in_x = x * self.strides[1]
                    in_slice = self.in_fmap[:, :, in_y:in_y+self.kernel_size[0], in_x:in_x+self.kernel_size[1]]
                    _map = in_slice * self.params['w1'][fi] + self.params['b1'][fi]
                    _map = _map.reshape(_map.shape[0], -1)
                    _red = np.sum(_map, axis=1)
                    self.output[:, fi, y, x] = _red

    def backward(self):
        # Add padding to the delta, to simplify code
        self.parents[0].delta = np.zeros_like(self.in_fmap)

        for fi in range(self.oshape[0]):  # For each filter
            for y in range(self.oshape[1]):  # Walk output's height
                in_y = y * self.strides[0]
                for x in range(self.oshape[2]):  # Walk output's width
                    in_x = x * self.strides[1]

                    # Add dL/dX (of window)
                    dhi = self.delta[:, fi, y, x]
                    w1_fi = self.params['w1'][fi]
                    dx = np.outer(dhi, w1_fi).reshape((len(dhi), *w1_fi.shape))
                    self.parents[0].delta[:, :, in_y:in_y+self.kernel_size[0], in_x:in_x+self.kernel_size[1]] += dx

                    # Get X (of window)
                    in_slice = self.in_fmap[:, :, in_y:in_y+self.kernel_size[0], in_x:in_x+self.kernel_size[1]]

                    # Compute gradients
                    g_w1 = np.mean(dhi.reshape((len(dhi), 1)) * in_slice.reshape((len(in_slice), -1)), axis=0).reshape(in_slice.shape[1:])
                    g_b1 = np.mean(dhi, axis=0)  #* 1.0

                    # Add regularizers (if needed)
                    if self.kernel_regularizer:
                        g_w1 += self.kernel_regularizer.backward(self.params['w1'][fi])
                    if self.bias_regularizer:
                        g_b1 += self.bias_regularizer.backward(self.params['b1'][fi])

                    self.grads["w1"][fi] += g_w1
                    self.grads["b1"][fi] += g_b1

        # Remove padding from delta
        self.parents[0].delta = self.parents[0].delta[:, :, self.pad_top:(self.shape_pad_in[1]-self.pad_bottom), self.pad_left:(self.shape_pad_in[2]-self.pad_right)]


class MaxPool(Layer):

    def __init__(self, l_in, pool_size, strides=(2, 2), padding="none", name="MaxPool"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Check layer compatibility
        if len(l_in.oshape) != 3:
            raise ValueError(f"Expected a 3D layer ({self.name})")

        # Params
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        # Compute sizes
        # Note: First compute the output size, then the padding (ignore the typical formula)
        _oshape = utils.get_output(input_size=l_in.oshape[1:], kernel_size=pool_size,
                                   strides=strides, padding=padding, dilation_rate=None)
        _pads = utils.get_padding(padding=padding, input_size=l_in.oshape[1:], output_size=_oshape[1:],
                                  kernel_size=pool_size, strides=strides)
        # Output / Pads size
        self.pads = tuple(_pads)  # height, width
        self.oshape = tuple([l_in.oshape[0]] + _oshape.tolist())

        # Specific pads
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = utils.get_padding_tblr(self.pads)

        # Caches
        self.output_idxs = None

    def forward(self):
        m = self.parents[0].output.shape[0]  # Batches
        self.output = np.zeros((m, *self.oshape))
        self.output_idxs = np.zeros_like(self.output).astype(int)

        # Reshape input adding paddings
        self.shape_pad_in = tuple(np.array(self.parents[0].oshape) + np.array([0, *self.pads]))
        self.in_fmap = np.zeros((m, *self.shape_pad_in))
        self.in_fmap[:, :, self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
        self.pad_left:(self.shape_pad_in[2] - self.pad_right)] = self.parents[0].output

        # Cache indexes
        for z in range(self.oshape[0]):  # For depth
            for y in range(self.oshape[1]):  # Walk output's height
                in_y = y * self.strides[0]
                for x in range(self.oshape[2]):  # Walk output's width
                    in_x = x * self.strides[1]
                    in_slice = self.in_fmap[:, z, in_y:in_y + self.pool_size[0], in_x:in_x + self.pool_size[1]]
                    in_slice = in_slice.reshape((len(in_slice), -1))
                    self.output[:, z, y, x] = np.amax(in_slice, axis=1)
                    self.output_idxs[:, z, y, x] = np.argmax(in_slice, axis=1)

    def backward(self):
        # Add padding to the delta, to simplify code
        self.parents[0].delta = np.zeros_like(self.in_fmap)

        for z in range(self.oshape[0]):  # For depth
            for y in range(self.oshape[1]):  # Walk output's height
                in_y = y * self.strides[0]
                for x in range(self.oshape[2]):  # Walk output's width
                    in_x = x * self.strides[1]

                    self.parents[0].delta[:, z, in_y:in_y + self.pool_size[0], in_x:in_x + self.pool_size[1]].flat[self.output_idxs[:, z, y, x]] += self.delta[:, z, y, x]

        # Remove padding from delta
        self.parents[0].delta = self.parents[0].delta[:, :, self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
                                self.pad_left:(self.shape_pad_in[2] - self.pad_right)]


class AvgPool(Layer):

    def __init__(self, l_in, pool_size, strides=(2, 2), padding="none", name="AvgPool"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Check layer compatibility
        if len(l_in.oshape) != 3:
            raise ValueError(f"Expected a 3D layer ({self.name})")

        # Params
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        # Compute sizes
        # Note: First compute the output size, then the padding (ignore the typical formula)
        _oshape = utils.get_output(input_size=l_in.oshape[1:], kernel_size=pool_size,
                                   strides=strides, padding=padding, dilation_rate=None)
        _pads = utils.get_padding(padding=padding, input_size=l_in.oshape[1:], output_size=_oshape[1:],
                                  kernel_size=pool_size, strides=strides)
        # Output / Pads size
        self.pads = tuple(_pads)  # height, width
        self.oshape = tuple([l_in.oshape[0]] + _oshape.tolist())

        # Specific pads
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = utils.get_padding_tblr(self.pads)

    def forward(self):
        m = self.parents[0].output.shape[0]  # Batches
        self.output = np.zeros((m, *self.oshape))

        # Reshape input adding paddings
        self.shape_pad_in = tuple(np.array(self.parents[0].oshape) + np.array([0, *self.pads]))
        self.in_fmap = np.zeros((m, *self.shape_pad_in))
        self.in_fmap[:, :, self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
        self.pad_left:(self.shape_pad_in[2] - self.pad_right)] = self.parents[0].output

        # Cache indexes
        for z in range(self.oshape[0]):  # For depth
            for y in range(self.oshape[1]):  # Walk output's height
                in_y = y * self.strides[0]
                for x in range(self.oshape[2]):  # Walk output's width
                    in_x = x * self.strides[1]
                    in_slice = self.in_fmap[:, z, in_y:in_y + self.pool_size[0], in_x:in_x + self.pool_size[1]]
                    in_slice = in_slice.reshape((len(in_slice), -1))
                    self.output[:, z, y, x] = np.mean(in_slice, axis=1)

    def backward(self):
        # Add padding to the delta, to simplify code
        self.parents[0].delta = np.zeros_like(self.in_fmap)

        psize = np.prod(self.pool_size)
        for z in range(self.oshape[0]):  # For depth
            for y in range(self.oshape[1]):  # Walk output's height
                in_y = y * self.strides[0]
                for x in range(self.oshape[2]):  # Walk output's width
                    in_x = x * self.strides[1]

                    dx = self.delta[:, z, y, x] * 1/psize
                    dx = np.tile(dx[:, np.newaxis], (1, 4)).reshape((-1, *self.pool_size))
                    self.parents[0].delta[:, z, in_y:in_y + self.pool_size[0], in_x:in_x + self.pool_size[1]] = dx

        # Remove padding from delta
        self.parents[0].delta = self.parents[0].delta[:, :, self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
                                self.pad_left:(self.shape_pad_in[2] - self.pad_right)]


class GlobalMaxPool(MaxPool):

    def __init__(self, l_in, name="GlobalMaxPool"):
        pool_size = l_in.oshape[1:]
        super().__init__(l_in, pool_size=pool_size, strides=(1, 1), padding="none", name=name)


class GlobalAvgPool(AvgPool):

    def __init__(self, l_in, name="GlobalAvgPool"):
        pool_size = l_in.oshape[1:]
        super().__init__(l_in, pool_size=pool_size, strides=(1, 1), padding="none", name=name)


class Embedding(Layer):
    """
    input_dim: vocabulary size
    output_dim: embedding latent space
    input_length: length of each sentence
    """
    def __init__(self, l_in, input_dim, output_dim, input_length, embeddings_initializer=None, embeddings_regularizer=None, name="Embedding"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.oshape = (input_length, output_dim)

        # Params and grads
        self.params = {'w1': np.zeros((input_dim, output_dim))}
        self.grads = {'w1': np.zeros_like(self.params['w1'])}

        # Initialization: param
        if embeddings_initializer is None:
            self.embeddings_initializer = initializers.RandomUniform()
        else:
            self.embeddings_initializer = embeddings_initializer

        # Add regularizers
        self.embeddings_regularizer = embeddings_regularizer

    def initialize(self, optimizer=None):
        super().initialize(optimizer=optimizer)

        # Initialize params
        self.embeddings_initializer.apply(self.params, ['w1'])

    def forward(self):
        # Get embeddings corresponding to word indices (for each sample)
        word_indices = self.parents[0].output
        sentence = self.params['w1'][word_indices]
        self.output = sentence  # (batch_size, input_length, self.output_dim)

    def backward(self):
        # Update embeddings corresponding to word indices
        word_indices = self.parents[0].output
        self.grads['w1'][word_indices] += np.mean(self.delta, axis=0)
