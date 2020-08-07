import copy
import numpy as np
from dnpy.layers import Layer
from dnpy import initializers, utils


class Pool(Layer):

    def __init__(self, l_in, pool_size, strides, padding, name="Pool"):
        super().__init__(name=name)
        self.parents.append(l_in)

        input_size = l_in.oshape[1:]

        # Params
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        # Compute sizes
        # Note: First compute the output size, then the padding (ignore the typical formula)
        _oshape = utils.get_output(input_size=input_size, kernel_size=pool_size,
                                   strides=strides, padding=padding, dilation_rate=None)
        _pads = utils.get_padding(padding=padding, input_size=input_size, output_size=_oshape[1:],
                                  kernel_size=pool_size, strides=strides)
        # Output / Pads size
        self.pads = tuple(_pads)  # height, width
        self.oshape = tuple([l_in.oshape[0]] + _oshape.tolist())


class MaxPool(Pool):

    def __init__(self, l_in, pool_size, strides=(2, 2), padding="none", name="MaxPool"):
        super().__init__(l_in, pool_size=pool_size, strides=strides,
                         padding=padding, name=name)

        # Specific pads
        (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right) = utils.get_side_paddings(self.pads)

        # Caches
        self.output_idxs = None

    def forward(self):
        batch_size = self.parents[0].output.shape[0]
        self.output = np.zeros((batch_size, *self.oshape))

        # Get kernel sizes
        (py, px) = self.pool_size
        (sy, sx) = self.strides
        (oz, oy, ox) = self.oshape

        # Reshape input adding paddings
        self.shape_pad_in = tuple(np.array(self.parents[0].oshape) + np.array([0, *self.pads]))
        self.in_fmap = np.zeros((batch_size, *self.shape_pad_in))
        self.in_fmap[:, :,
        self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
        self.pad_left:(self.shape_pad_in[2] - self.pad_right)] = self.parents[0].output

        # Cache indexes
        self.output_idxs = np.zeros_like(self.output).astype(int)

        for z in range(oz):  # For depth

            for y in range(oy):  # Walk output's height
                in_y = y * sy

                for x in range(ox):  # Walk output's width
                    in_x = x * sx

                    # Get slice
                    in_slice = self.in_fmap[:, z, in_y:in_y + py, in_x:in_x + px]

                    # Perform pooling
                    in_slice = in_slice.reshape((len(in_slice), -1))
                    self.output[:, z, y, x] = np.amax(in_slice, axis=1)
                    self.output_idxs[:, z, y, x] = np.argmax(in_slice, axis=1)

    def backward(self):
        # Add padding to the delta, to simplify code
        self.parents[0].delta = np.zeros_like(self.in_fmap)

        # Get kernel sizes
        (py, px) = self.pool_size
        (sy, sx) = self.strides
        (oz, oy, ox) = self.oshape

        for z in range(oz):  # For depth

            for y in range(oy):  # Walk output's height
                in_y = y * sy

                for x in range(ox):  # Walk output's width
                    in_x = x * sx

                    self.parents[0].delta[:, z, in_y:in_y + py, in_x:in_x + px].flat[self.output_idxs[:, z, y, x]] += self.delta[:, z, y, x]

        # Remove padding from delta
        self.parents[0].delta = self.parents[0].delta[:, :, self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
                                self.pad_left:(self.shape_pad_in[2] - self.pad_right)]


class GlobalMaxPool(MaxPool):

    def __init__(self, l_in, name="GlobalMaxPool"):
        pool_size = l_in.oshape[1:]
        super().__init__(l_in, pool_size=pool_size, strides=(1, 1), padding="none", name=name)


class AvgPool(Pool):

    def __init__(self, l_in, pool_size, strides=(2, 2), padding="none", name="AvgPool"):
        super().__init__(l_in, pool_size=pool_size, strides=strides,
                         padding=padding, name=name)

        # Specific pads
        (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right) = utils.get_side_paddings(self.pads)

    def forward(self):
        batch_size = self.parents[0].output.shape[0]
        self.output = np.zeros((batch_size, *self.oshape))

        # Get kernel sizes
        (py, px) = self.pool_size
        (sy, sx) = self.strides
        (oz, oy, ox) = self.oshape

        # Reshape input adding paddings
        self.shape_pad_in = tuple(np.array(self.parents[0].oshape) + np.array([0, *self.pads]))
        self.in_fmap = np.zeros((batch_size, *self.shape_pad_in))
        self.in_fmap[:, :,
        self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
        self.pad_left:(self.shape_pad_in[2] - self.pad_right)] = self.parents[0].output

        for z in range(oz):  # For depth

            for y in range(oy):  # Walk output's height
                in_y = y * sy

                for x in range(ox):  # Walk output's width
                    in_x = x * sx

                    # Get slice
                    in_slice = self.in_fmap[:, z, in_y:in_y + py, in_x:in_x + px]

                    # Perform pooling
                    in_slice = in_slice.reshape((len(in_slice), -1))
                    self.output[:, z, y, x] = np.mean(in_slice, axis=1)

    def backward(self):
        # Add padding to the delta, to simplify code
        self.parents[0].delta = np.zeros_like(self.in_fmap)

        # Get kernel sizes
        (py, px) = self.pool_size
        (sy, sx) = self.strides
        (oz, oy, ox) = self.oshape

        psize = np.prod(self.pool_size)
        for z in range(oz):  # For depth

            for y in range(oy):  # Walk output's height
                in_y = y * sy

                for x in range(ox):  # Walk output's width
                    in_x = x * sx

                    # Perform back-pooling
                    dx = self.delta[:, z, y, x] * 1/psize
                    dx = np.tile(dx[:, np.newaxis], (1, 4)).reshape((-1, *self.pool_size))
                    self.parents[0].delta[:, z, in_y:in_y + py, in_x:in_x + px] = dx

        # Remove padding from delta
        self.parents[0].delta = self.parents[0].delta[:, :, self.pad_top:(self.shape_pad_in[1] - self.pad_bottom),
                                self.pad_left:(self.shape_pad_in[2] - self.pad_right)]


class GlobalAvgPool(AvgPool):

    def __init__(self, l_in, name="GlobalAvgPool"):
        pool_size = l_in.oshape[1:]
        super().__init__(l_in, pool_size=pool_size, strides=(1, 1), padding="none", name=name)
