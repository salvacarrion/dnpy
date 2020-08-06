import copy
import numpy as np
from dnpy.layers import Layer
from dnpy import initializers, utils


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


class GlobalMaxPool(MaxPool):

    def __init__(self, l_in, name="GlobalMaxPool"):
        pool_size = l_in.oshape[1:]
        super().__init__(l_in, pool_size=pool_size, strides=(1, 1), padding="none", name=name)


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


class GlobalAvgPool(AvgPool):

    def __init__(self, l_in, name="GlobalAvgPool"):
        pool_size = l_in.oshape[1:]
        super().__init__(l_in, pool_size=pool_size, strides=(1, 1), padding="none", name=name)
