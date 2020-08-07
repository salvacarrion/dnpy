import copy
import numpy as np
from dnpy.layers import Layer
from dnpy import initializers, utils


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
            # # There are "num input feature maps * filter height * filter width" inputs to each hidden unit
            # out_fm, in_fm, f_height, f_width = self.params['w1'].shape
            # fan_in = in_fm*f_height*f_width
            #
            # #  Each unit in the lower layer receives a gradient from:
            # #  "num output feature maps * filter height * filter width" / pooling size
            # fan_out = out_fm*f_height*f_width
            # self.kernel_initializer = initializers.HeNormal(fan_in=fan_in, fan_out=fan_out)
            self.kernel_initializer = initializers.RandomUniform()
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

        for fi in range(self.oshape[0]):  # For each filter (independent of the image)
            for y in range(self.oshape[1]):  # Walk output's height
                in_y = y * self.strides[0]
                for x in range(self.oshape[2]):  # Walk output's width
                    in_x = x * self.strides[1]
                    in_slice = self.in_fmap[:, :, in_y:in_y+self.kernel_size[0], in_x:in_x+self.kernel_size[1]]
                    _map = in_slice * self.params['w1'][fi] + self.params['b1'][fi]
                    _red = np.sum(_map, axis=(1, 2, 3))
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
                    g_w1 = np.mean(np.expand_dims(dhi, axis=(1, 2, 3)) * in_slice, axis=0)
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


class PointwiseConv2D(Conv2D):

    def __init__(self, l_in, filters,
                 kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None, name="PointwiseConv2D"):
        super().__init__(l_in, filters, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding="none",
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         name=name)


class DepthwiseConv2D(Layer):

    def __init__(self, l_in, kernel_size, strides=(1, 1), padding="none",
                 dilation_rate=(1, 1), depth_multiplier=1,
                 kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None, name="DepthwiseConv2D"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Check layer compatibility
        if len(l_in.oshape) != 3:
            raise ValueError(f"Expected a 3D layer ({self.name})")

        # Params
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.depth_multiplier = int(depth_multiplier)

        # Compute sizes
        # Note: First compute the output size, then the padding (ignore the typical formula)
        channels = l_in.oshape[0]
        _oshape = utils.get_output(input_size=l_in.oshape[1:], kernel_size=kernel_size,
                                   strides=strides, padding=padding, dilation_rate=dilation_rate)
        _pads = utils.get_padding(padding=padding, input_size=l_in.oshape[1:], output_size=_oshape[1:],
                                  kernel_size=kernel_size, strides=strides)
        # Output / Pads size
        self.pads = tuple(_pads)  # height, width
        self.oshape = tuple([channels*self.depth_multiplier] + _oshape.tolist())

        # Specific pads
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = utils.get_padding_tblr(self.pads)

        # Params and grads
        self.params = {'w1': np.zeros((1, channels, *self.kernel_size)),
                       'b1': np.zeros((channels,))}  # One filter per channel
        self.grads = {'w1': np.zeros_like(self.params['w1']),
                      'b1': np.zeros_like(self.params['b1'])}

        # Initialization: param
        if kernel_initializer is None:
            # # There are "num input feature maps * filter height * filter width" inputs to each hidden unit
            # out_fm, in_fm, f_height, f_width = self.params['w1'].shape
            # fan_in = in_fm*f_height*f_width
            #
            # #  Each unit in the lower layer receives a gradient from:
            # #  "num output feature maps * filter height * filter width" / pooling size
            # fan_out = out_fm*f_height*f_width
            # self.kernel_initializer = initializers.HeNormal(fan_in=fan_in, fan_out=fan_out)
            self.kernel_initializer = initializers.RandomUniform()
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

        for y in range(self.oshape[1]):  # Walk output's height
            in_y = y * self.strides[0]
            for x in range(self.oshape[2]):  # Walk output's width
                in_x = x * self.strides[1]
                in_slice = self.in_fmap[:, :, in_y:in_y+self.kernel_size[0], in_x:in_x+self.kernel_size[1]]
                _map = in_slice * self.params['w1'] + self.params['b1']
                _red = np.sum(_map, axis=(2, 3))
                self.output[:, :, y, x] = _red

    def backward(self):
        # Add padding to the delta, to simplify code
        self.parents[0].delta = np.zeros_like(self.in_fmap)

        for y in range(self.oshape[1]):  # Walk output's height
            in_y = y * self.strides[0]
            for x in range(self.oshape[2]):  # Walk output's width
                in_x = x * self.strides[1]

                # Add dL/dX (of window)
                dhi = self.delta[:, :, y, x]
                dx = np.expand_dims(dhi, axis=(2, 3)) * self.params['w1']
                self.parents[0].delta[:, :, in_y:in_y+self.kernel_size[0], in_x:in_x+self.kernel_size[1]] += dx

                # Get X (of window)
                in_slice = self.in_fmap[:, :, in_y:in_y+self.kernel_size[0], in_x:in_x+self.kernel_size[1]]

                # Compute gradients
                g_w1 = np.mean(np.expand_dims(dhi, axis=(2, 3)) * in_slice, axis=0)
                g_b1 = np.mean(dhi, axis=0)  #* 1.0

                # Add regularizers (if needed)
                if self.kernel_regularizer:
                    g_w1 += self.kernel_regularizer.backward(self.params['w1'])
                if self.bias_regularizer:
                    g_b1 += self.bias_regularizer.backward(self.params['b1'])

                self.grads["w1"] += g_w1
                self.grads["b1"] += g_b1

        # Remove padding from delta
        self.parents[0].delta = self.parents[0].delta[:, :, self.pad_top:(self.shape_pad_in[1]-self.pad_bottom), self.pad_left:(self.shape_pad_in[2]-self.pad_right)]
