from dnpy import utils
from dnpy.layers import *

import numpy as np


def test_outputs():
    r1 = utils.get_output(input_size=(4, 4), kernel_size=(3, 3), strides=(1, 1), padding=(0, 0))
    assert np.all(r1 == np.array((2, 2)))

    r2 = utils.get_output(input_size=(5, 5), kernel_size=(4, 4), strides=(1, 1), padding=(2, 2))
    assert np.all(r2 == np.array((6, 6)))

    r3 = utils.get_output(input_size=(5, 5), kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    assert np.all(r3 == np.array((5, 5)))

    r4 = utils.get_output(input_size=(5, 5), kernel_size=(3, 3), strides=(1, 1), padding=(2, 2))
    assert np.all(r4 == np.array((7, 7)))

    r5 = utils.get_output(input_size=(5, 5), kernel_size=(3, 3), strides=(2, 2), padding=(0, 0))
    assert np.all(r5 == np.array((2, 2)))

    r6 = utils.get_output(input_size=(5, 5), kernel_size=(3, 3), strides=(2, 2), padding=(1, 1))
    assert np.all(r6 == np.array((3, 3)))

