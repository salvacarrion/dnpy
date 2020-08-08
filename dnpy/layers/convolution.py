import copy
import numpy as np
from dnpy.layers import Layer
from dnpy import initializers, utils


class Conv(Layer):

    def __init__(self, l_in, filters, kernel_size, strides, padding, dilation_rate,
                 kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None, name="Conv"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Infer input size
        if len(kernel_size) == len(l_in.oshape):
            input_size = l_in.oshape
            self.kernel_size = kernel_size
        else:
            input_size = l_in.oshape[1:]
            self.kernel_size = (l_in.oshape[0], *kernel_size)

        # Params
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate

        # Compute sizes
        # Note: First compute the output size, then the padding (ignore the typical formula)
        _oshape = utils.get_output(input_size=input_size, kernel_size=kernel_size,
                                   strides=strides, padding=padding, dilation_rate=dilation_rate)
        _pads = utils.get_padding(padding=padding, input_size=input_size, output_size=_oshape,
                                  kernel_size=kernel_size, strides=strides)
        # Output / Pads size
        self.pads = tuple(_pads)  # height, width
        self.oshape = tuple([self.filters] + _oshape.tolist())

        # Params and grads
        self.params = {'w1': np.zeros((self.filters, *self.kernel_size)),
                       'b1': np.zeros((self.filters,))}
        self.grads = {'w1': np.zeros_like(self.params['w1']),
                      'b1': np.zeros_like(self.params['b1'])}

        # Initialization: param
        if kernel_initializer is None:
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


class Conv2D(Conv):

    def __init__(self, l_in, filters, kernel_size, strides=(1, 1), padding="same",
                 dilation_rate=(1, 1),
                 kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None, name="Conv2D"):

        super().__init__(l_in, filters=filters, kernel_size=kernel_size, strides=strides,
                         padding=padding, dilation_rate=dilation_rate,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name=name)

        # Specific pads
        (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right) = utils.get_side_paddings(self.pads)

        assert len(self.kernel_size) == 3
        assert len(self.strides) == 2

    def forward(self):
        batch_size = self.parents[0].output.shape[0]
        self.output = np.zeros((batch_size, *self.oshape))

        # Get kernel sizes
        (kz, ky, kx) = self.kernel_size
        (sy, sx) = self.strides
        (of, oz, oy, ox) = (self.filters, *self.oshape)

        # Reshape input adding paddings
        self.shape_pad_in = tuple(np.array(self.parents[0].oshape) + np.array([0, *self.pads]))
        self.in_fmap = np.zeros((batch_size, *self.shape_pad_in))
        self.in_fmap[:, :,
        self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
        self.pad_left:(self.shape_pad_in[2] - self.pad_right)] = self.parents[0].output

        for f in range(of):  # For each filter

                for y in range(oy):  # Walk output's height
                    in_y = y * sy

                    for x in range(ox):  # Walk output's width
                        in_x = x * sx

                        # Get slice
                        in_slice = self.in_fmap[:, :,
                                   in_y:in_y + ky,
                                   in_x:in_x + kx]

                        # Perform convolution
                        _map = in_slice * self.params['w1'][f] + self.params['b1'][f]
                        _red = np.sum(_map, axis=(1, 2, 3))
                        self.output[:, f, y, x] = _red

    def backward(self):
        # Add padding to the delta, to simplify code
        self.parents[0].delta = np.zeros_like(self.in_fmap)

        # Get kernel sizes
        (kz, ky, kx) = self.kernel_size
        (sy, sx) = (self.strides)
        (of, oz, oy, ox) = (self.filters, *self.oshape)

        for f in range(of):  # For each filter

                for y in range(oy):  # Walk output's height
                    in_y = y * sy

                    for x in range(ox):  # Walk output's width
                        in_x = x * sx

                        # Add dL/dX (of window)
                        dhi = self.delta[:, f, y, x]
                        w1_f = self.params['w1'][f]
                        dx = np.outer(dhi, w1_f).reshape((len(dhi), *w1_f.shape))
                        self.parents[0].delta[:, :,
                        in_y:in_y + ky,
                        in_x:in_x + kx] += dx

                        # Get X (of window)
                        in_slice = self.in_fmap[:, :,
                                   in_y:in_y + ky,
                                   in_x:in_x + kx]

                        # Compute gradients
                        g_w1 = np.mean(np.expand_dims(dhi, axis=(1, 2, 3)) * in_slice, axis=0)
                        g_b1 = np.mean(dhi, axis=0)  # * 1.0

                        # Add regularizers (if needed)
                        if self.kernel_regularizer:
                            g_w1 += self.kernel_regularizer.backward(self.params['w1'][f])
                        if self.bias_regularizer:
                            g_b1 += self.bias_regularizer.backward(self.params['b1'][f])

                        self.grads["w1"][f] += g_w1
                        self.grads["b1"][f] += g_b1

        # Remove padding from delta
        self.parents[0].delta = self.parents[0].delta[:, :,
                                self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
                                self.pad_left:(self.shape_pad_in[2] - self.pad_right)]


class Conv2D(Conv):

    def __init__(self, l_in, filters, kernel_size, strides=(1, 1), padding="same",
                 dilation_rate=(1, 1),
                 kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None, name="Conv2D"):

        super().__init__(l_in, filters=filters, kernel_size=kernel_size, strides=strides,
                         padding=padding, dilation_rate=dilation_rate,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name=name)

        # Specific pads
        (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right) = utils.get_side_paddings(self.pads)

        assert len(self.kernel_size) == 3
        assert len(self.strides) == 2

    def forward(self):
        batch_size = self.parents[0].output.shape[0]
        self.output = np.zeros((batch_size, *self.oshape))

        # Get kernel sizes
        (kz, ky, kx) = self.kernel_size
        (sy, sx) = self.strides
        (of, oz, oy, ox) = (self.filters, *self.oshape)

        # Reshape input adding paddings
        self.shape_pad_in = tuple(np.array(self.parents[0].oshape) + np.array([0, *self.pads]))
        self.in_fmap = np.zeros((batch_size, *self.shape_pad_in))
        self.in_fmap[:, :,
        self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
        self.pad_left:(self.shape_pad_in[2] - self.pad_right)] = self.parents[0].output

        for f in range(of):  # For each filter

                for y in range(oy):  # Walk output's height
                    in_y = y * sy

                    for x in range(ox):  # Walk output's width
                        in_x = x * sx

                        # Get slice
                        in_slice = self.in_fmap[:, :,
                                   in_y:in_y + ky,
                                   in_x:in_x + kx]

                        # Perform convolution
                        _map = in_slice * self.params['w1'][f] + self.params['b1'][f]
                        _red = np.sum(_map, axis=(1, 2, 3))
                        self.output[:, f, y, x] = _red

    def backward(self):
        # Add padding to the delta, to simplify code
        self.parents[0].delta = np.zeros_like(self.in_fmap)

        # Get kernel sizes
        (kz, ky, kx) = self.kernel_size
        (sy, sx) = (self.strides)
        (of, oz, oy, ox) = (self.filters, *self.oshape)

        for f in range(of):  # For each filter

                for y in range(oy):  # Walk output's height
                    in_y = y * sy

                    for x in range(ox):  # Walk output's width
                        in_x = x * sx

                        # Add dL/dX (of window)
                        dhi = self.delta[:, f, y, x]
                        w1_f = self.params['w1'][f]
                        dx = np.outer(dhi, w1_f).reshape((len(dhi), *w1_f.shape))
                        self.parents[0].delta[:, :,
                        in_y:in_y + ky,
                        in_x:in_x + kx] += dx

                        # Get X (of window)
                        in_slice = self.in_fmap[:, :,
                                   in_y:in_y + ky,
                                   in_x:in_x + kx]

                        # Compute gradients
                        g_w1 = np.mean(np.expand_dims(dhi, axis=(1, 2, 3)) * in_slice, axis=0)
                        g_b1 = np.mean(dhi, axis=0)  # * 1.0

                        # Add regularizers (if needed)
                        if self.kernel_regularizer:
                            g_w1 += self.kernel_regularizer.backward(self.params['w1'][f])
                        if self.bias_regularizer:
                            g_b1 += self.bias_regularizer.backward(self.params['b1'][f])

                        self.grads["w1"][f] += g_w1
                        self.grads["b1"][f] += g_b1

        # Remove padding from delta
        self.parents[0].delta = self.parents[0].delta[:, :,
                                self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
                                self.pad_left:(self.shape_pad_in[2] - self.pad_right)]


class PointwiseConv2D(Conv2D):

    def __init__(self, l_in, filters,
                 kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None, name="PointwiseConv2D"):
        super().__init__(l_in, filters, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding="none",
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         name=name)


class DepthwiseConv2D(Conv):

    def __init__(self, l_in, kernel_size, strides=(1, 1), padding="same",
                 dilation_rate=(1, 1), depth_multiplier=1,
                 kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None, name="DepthwiseConv2D"):

        # Check layer compatibility
        if len(l_in.oshape) != 3:
            raise ValueError(f"Expected a 3D layer ({self.name})")

        filters = l_in.oshape[0]
        self.depth_multiplier = int(depth_multiplier)

        super().__init__(l_in, filters=filters, kernel_size=kernel_size, strides=strides,
                         padding=padding, dilation_rate=dilation_rate,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name=name)

        # Specific pads
        (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right) = utils.get_side_paddings(self.pads)

        # Overwrite stuff
        assert len(self.kernel_size) == 3
        assert len(self.strides) == 2

        # Params and grads
        self.kernel_size = (1, *list(self.kernel_size)[1:])  # Depth-wise
        self.params = {'w1': np.zeros((self.filters, *self.kernel_size)),
                       'b1': np.zeros((self.filters,))}
        self.grads = {'w1': np.zeros_like(self.params['w1']),
                      'b1': np.zeros_like(self.params['b1'])}

    def forward(self):
        batch_size = self.parents[0].output.shape[0]
        self.output = np.zeros((batch_size, *self.oshape))

        # Get kernel sizes
        (kz, ky, kx) = self.kernel_size
        (sy, sx) = self.strides
        (of, oz, oy, ox) = (self.filters, *self.oshape)

        # Reshape input adding paddings
        self.shape_pad_in = tuple(np.array(self.parents[0].oshape) + np.array([0, *self.pads]))
        self.in_fmap = np.zeros((batch_size, *self.shape_pad_in))
        self.in_fmap[:, :,
        self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
        self.pad_left:(self.shape_pad_in[2] - self.pad_right)] = self.parents[0].output

        for f in range(of):  # For each filter

            for y in range(oy):  # Walk output's height
                in_y = y * sy

                for x in range(ox):  # Walk output's width
                    in_x = x * sx

                    in_slice = self.in_fmap[:, f, in_y:in_y+ky, in_x:in_x+kx]
                    _map = in_slice * self.params['w1'][f] + self.params['b1'][f]
                    _red = np.sum(_map, axis=(1, 2))
                    self.output[:, f, y, x] = _red

    def backward(self):
        # Add padding to the delta, to simplify code
        self.parents[0].delta = np.zeros_like(self.in_fmap)

        # Get kernel sizes
        (kz, ky, kx) = self.kernel_size
        (sy, sx) = (self.strides)
        (of, oz, oy, ox) = (self.filters, *self.oshape)

        for f in range(of):  # For each filter

            for y in range(oy):  # Walk output's height
                in_y = y * sy

                for x in range(ox):  # Walk output's width
                    in_x = x * sx

                    # Add dL/dX (of window)
                    dhi = self.delta[:, f, y, x]
                    dx = np.expand_dims(dhi, axis=(1, 2)) * self.params['w1'][f]
                    self.parents[0].delta[:, f, in_y:in_y+ky, in_x:in_x+kx] += dx

                    # Get X (of window)
                    in_slice = self.in_fmap[:, f, in_y:in_y+ky, in_x:in_x+kx]

                    # Compute gradients
                    g_w1 = np.mean(np.expand_dims(dhi, axis=(1, 2)) * in_slice, axis=0)
                    g_b1 = np.mean(dhi, axis=0)  #* 1.0

                    # Add regularizers (if needed)
                    if self.kernel_regularizer:
                        g_w1 += self.kernel_regularizer.backward(self.params['w1'][f])
                    if self.bias_regularizer:
                        g_b1 += self.bias_regularizer.backward(self.params['b1'][f])

                    self.grads["w1"][f] += g_w1
                    self.grads["b1"][f] += g_b1

        # Remove padding from delta
        self.parents[0].delta = self.parents[0].delta[:, :, self.pad_top:(self.shape_pad_in[1]-self.pad_bottom), self.pad_left:(self.shape_pad_in[2]-self.pad_right)]
