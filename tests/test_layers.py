import unittest

from dnpy import utils
from dnpy.layers import *
from dnpy.net import Net
from dnpy.optimizers import *

import numpy as np


class TestStringMethods(unittest.TestCase):

    def test_outputs(self):
        r1 = utils.get_output(input_size=(4, 4), kernel_size=(3, 3), strides=(1, 1), padding=(0, 0))
        self.assertTrue(np.all(r1 == np.array((2, 2))))

        r2 = utils.get_output(input_size=(5, 5), kernel_size=(4, 4), strides=(1, 1), padding=(2, 2))
        self.assertTrue(np.all(r2 == np.array((6, 6))))

        r3 = utils.get_output(input_size=(5, 5), kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
        self.assertTrue(np.all(r3 == np.array((5, 5))))

        r4 = utils.get_output(input_size=(5, 5), kernel_size=(3, 3), strides=(1, 1), padding=(2, 2))
        self.assertTrue(np.all(r4 == np.array((7, 7))))

        r5 = utils.get_output(input_size=(5, 5), kernel_size=(3, 3), strides=(2, 2), padding=(0, 0))
        self.assertTrue(np.all(r5 == np.array((2, 2))))

        r6 = utils.get_output(input_size=(5, 5), kernel_size=(3, 3), strides=(2, 2), padding=(1, 1))
        self.assertTrue(np.all(r6 == np.array((3, 3))))

    def test_maxpool(self):
        # Test 1
        t1_in_img = np.array([[
            [12, 20, 30, 0],
            [8, 12, 2, 0],
            [34, 70, 37, 4],
            [112, 100, 25, 12],
        ]])
        t1_ref_img = np.array([[
            [20, 30],
            [112, 37]
        ]])

        # Test 1
        l_in = Input(shape=t1_in_img.shape)
        l_in.output = t1_in_img
        l1 = MaxPool(l_in, pool_size=(2, 2), strides=(2, 2), padding="none")
        l1.forward()
        self.assertTrue(np.all(t1_ref_img == l1.output))

        dfas = 3


if __name__ == "__main__":
    unittest.main()
