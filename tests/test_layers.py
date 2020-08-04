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

    def test_conv2d_2x2_none_temp1(self):
        # Test 1
        t1_in_img = np.array([[
            [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 4, 5, 0],
            [0, 2, 3, 2, 1, 3, 0],
            [0, 4, 4, 0, 4, 3, 0],
            [0, 2, 5, 2, 6, 4, 0],
            [0, 1, 0, 0, 5, 7, 0],
            [0, 0, 0, 0, 0, 0, 0],
            ]
        ]])

        t1_ref_img = np.array([[
            [
                [0, 1, 1, 4,  9, 5],
                [2, 6, 6, 7, 13, 8],
                [6, 13, 9, 7, 11, 6],
                [6, 15, 11, 12, 17, 7],
                [3, 8, 7, 13, 22, 11],
                [1, 1, 0, 5, 12, 7]
            ]
        ]])

        t1_ref_back = np.array([[
            [
                [1, 2, 2, 2, 2, 2, 1],
                [2, 4, 4, 4, 4, 4, 2],
                [2, 4, 4, 4, 4, 4, 2],
                [2, 4, 4, 4, 4, 4, 2],
                [2, 4, 4, 4, 4, 4, 2],
                [2, 4, 4, 4, 4, 4, 2],
                [1, 2, 2, 2, 2, 2, 1]
            ]
        ]])

        # Test 1
        # Forward
        l0 = Input(shape=t1_in_img[0].shape)
        l0.output = t1_in_img
        l1 = Conv2D(l0, 1, kernel_size=(2, 2), strides=(1, 1), padding="none",
                    kernel_initializer=initializers.Constant(fill_value=1))
        l1.initialize()
        l1.forward()
        self.assertTrue(np.all(t1_ref_img == l1.output))

        # Backward
        l1.delta = np.ones_like(t1_ref_img)
        l1.backward()
        self.assertTrue(np.all(t1_ref_back == l0.delta))

    def test_conv2d_2x2_none_temp2(self):
        # Test 1
        t1_in_img = np.array([[
            [
                [0, 1, 0, 4, 5, 0],
                [2, 3, 2, 1, 3, 0],
                [4, 4, 0, 4, 3, 0],
                [2, 5, 2, 6, 4, 0],
                [1, 0, 0, 5, 7, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ]])

        t1_ref_img = np.array([[
            [
                [6, 6, 7, 13, 8],
                [13, 9, 7, 11, 6],
                [15, 11, 12, 17, 7],
                [8, 7, 13, 22, 11],
                [1, 0, 5, 12, 7]
            ]
        ]])
        
        t1_ref_back = np.array([[
            [
                [1, 2, 2, 2, 2, 1],
                [2, 4, 4, 4, 4, 2],
                [2, 4, 4, 4, 4, 2],
                [2, 4, 4, 4, 4, 2],
                [2, 4, 4, 4, 4, 2],
                [1, 2, 2, 2, 2, 1]
            ]
        ]])

        # Test 1
        # Forward
        l0 = Input(shape=t1_in_img[0].shape)
        l0.output = t1_in_img
        l1 = Conv2D(l0, 1, kernel_size=(2, 2), strides=(1, 1), padding="none",
                    kernel_initializer=initializers.Constant(fill_value=1))
        l1.initialize()
        l1.forward()
        self.assertTrue(np.all(t1_ref_img == l1.output))

        # Backward
        l1.delta = np.ones_like(t1_ref_img)
        l1.backward()
        self.assertTrue(np.all(t1_ref_back == l0.delta))

    def test_conv2d_2x2_none(self):
        # Test 1
        t1_in_img = np.array([[
            [
            [0, 1, 0, 4, 5],
            [2, 3, 2, 1, 3],
            [4, 4, 0, 4, 3],
            [2, 5, 2, 6, 4],
            [1, 0, 0, 5, 7],
            ]
        ]])

        t1_ref_img = np.array([[
            [
             [6, 6, 7, 13],
             [13, 9, 7, 11],
             [15, 11, 12, 17],
             [8, 7, 13, 22]
            ]
        ]])

        t1_ref_back = np.array([[
            [
            [1, 2, 2, 2, 1],
            [2, 4, 4, 4, 2],
            [2, 4, 4, 4, 2],
            [2, 4, 4, 4, 2],
            [1, 2, 2, 2, 1],
            ]
        ]])

        # Test 1
        # Forward
        l0 = Input(shape=t1_in_img[0].shape)
        l0.output = t1_in_img
        l1 = Conv2D(l0, 1, kernel_size=(2, 2), strides=(1, 1), padding="none",
                    kernel_initializer=initializers.Constant(fill_value=1))
        l1.initialize()
        l1.forward()
        self.assertTrue(np.all(t1_ref_img == l1.output))

        # Backward
        l1.delta = np.ones_like(t1_ref_img)
        l1.backward()
        self.assertTrue(np.all(t1_ref_back == l0.delta))

    def test_conv2d_2x2_same(self):
        # Test 1
        t1_in_img = np.array([[
            [
            [0, 1, 0, 4, 5],
            [2, 3, 2, 1, 3],
            [4, 4, 0, 4, 3],
            [2, 5, 2, 6, 4],
            [1, 0, 0, 5, 7],
            ]
        ]])
        t1_ref_img = np.array([[
            [
                [6, 6, 7, 13, 8],
                [13, 9, 7, 11, 6],
                [15, 11, 12, 17, 7],
                [8, 7, 13, 22, 11],
                [1, 0, 5, 12, 7]
            ]
        ]])

        t1_ref_back = np.array([[
            [
            [1, 2, 2, 2, 2],
            [2, 4, 4, 4, 4],
            [2, 4, 4, 4, 4],
            [2, 4, 4, 4, 4],
            [2, 4, 4, 4, 4]
            ]
        ]])

        # Test 1
        # Forward
        l0 = Input(shape=t1_in_img[0].shape)
        l0.output = t1_in_img
        l1 = Conv2D(l0, 1, kernel_size=(2, 2), strides=(1, 1), padding="same",
                    kernel_initializer=initializers.Constant(fill_value=1))
        l1.initialize()
        l1.forward()
        self.assertTrue(np.all(t1_ref_img == l1.output))

        # Backward
        l1.delta = np.ones_like(t1_ref_img)
        l1.backward()
        self.assertTrue(np.all(t1_ref_back == l0.delta))

    def test_conv2d_3x3_same(self):
        # Test 1
        t1_in_img = np.array([[
            [
                [0, 1, 0, 4, 5],
                [2, 3, 2, 1, 3],
                [4, 4, 0, 4, 3],
                [2, 5, 2, 6, 4],
                [1, 0, 0, 5, 7],
            ]
        ]])
        t1_ref_img = np.array([[
            [
            [6, 8, 11, 15, 13],
            [14, 16, 19, 22, 20],
            [20, 24, 27, 25, 21],
            [16, 18, 26, 31, 29],
            [8, 10, 18, 24, 22],
            ]
        ]])

        t1_ref_back = np.array([[
            [
            [4, 6, 6, 6, 4],
            [6, 9, 9, 9, 6],
            [6, 9, 9, 9, 6],
            [6, 9, 9, 9, 6],
            [4, 6, 6, 6, 4],
            ]
        ]])

        # Test 1
        # Forward
        l0 = Input(shape=t1_in_img[0].shape)
        l0.output = t1_in_img
        l1 = Conv2D(l0, 1, kernel_size=(3, 3), strides=(1, 1), padding="same",
                    kernel_initializer=initializers.Constant(fill_value=1))
        l1.initialize()
        l1.forward()
        self.assertTrue(np.all(t1_ref_img == l1.output))

        # Backward
        l1.delta = np.ones_like(t1_ref_img)
        l1.backward()
        self.assertTrue(np.all(t1_ref_back == l0.delta))

    def test_maxpool_2x2_none(self):
        # Test 1
        t1_in_img = np.array([[
            [
            [0, 1, 0, 4, 5],
            [2, 3, 2, 1, 3],
            [4, 4, 0, 4, 3],
            [2, 5, 2, 6, 4],
            [1, 0, 0, 5, 7],
            ]
        ]])
        t1_ref_img = np.array([[
            [
            [3, 4],
            [5, 6],
            ]
        ]])

        t1_ref_back = np.array([[
            [
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            ]
        ]])

        # Test 1
        # Forward
        l0 = Input(shape=t1_in_img[0].shape)
        l0.output = t1_in_img
        l1 = MaxPool(l0, pool_size=(2, 2), strides=(2, 2), padding="none")
        l1.forward()
        self.assertTrue(np.all(t1_ref_img == l1.output))

        # Backward
        l1.delta = np.ones_like(t1_ref_img)
        l1.backward()
        self.assertTrue(np.all(t1_ref_back == l0.delta))

    def test_maxpool_2x2_same(self):
        # Test 1
        t1_in_img = np.array([[
            [
            [0, 1, 0, 4, 5],
            [2, 3, 2, 1, 3],
            [4, 4, 0, 4, 3],
            [2, 5, 2, 6, 4],
            [1, 0, 0, 5, 7],
            ]
        ]])
        t1_ref_img = np.array([[
            [
            [3, 4, 5],
            [5, 6, 4],
            [1, 5, 7],
            ]
        ]])

        t1_ref_back = np.array([[
            [
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1],
            [1, 0, 0, 1, 1],
            ]
        ]])

        # Test 1
        # Forward
        l0 = Input(shape=t1_in_img[0].shape)
        l0.output = t1_in_img
        l1 = MaxPool(l0, pool_size=(2, 2), strides=(2, 2), padding="same")
        l1.forward()
        self.assertTrue(np.all(t1_ref_img == l1.output))

        # Backward
        l1.delta = np.ones_like(t1_ref_img)
        l1.backward()
        self.assertTrue(np.all(t1_ref_back == l0.delta))

    def test_avgpool_2x2_none(self):
        # Test 1
        t1_in_img = np.array([[
            [
                [0, 1, 0, 4, 5],
                [2, 3, 2, 1, 3],
                [4, 4, 0, 4, 3],
                [2, 5, 2, 6, 4],
                [1, 0, 0, 5, 7],
            ]
        ]])
        t1_ref_img = np.array([[
            [
            [1.5, 1.75],
            [3.75, 3.0],
            ]
        ]])

        t1_ref_back = np.array([[
            [
             [0.25000, 0.25000, 0.25000, 0.25000, 0.00000],
             [0.25000, 0.25000, 0.25000, 0.25000, 0.00000],
             [0.25000, 0.25000, 0.25000, 0.25000, 0.00000],
             [0.25000, 0.25000, 0.25000, 0.25000, 0.00000],
             [0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
            ]
        ]])

        # Test 1
        # Forward
        l0 = Input(shape=t1_in_img[0].shape)
        l0.output = t1_in_img
        l1 = AvgPool(l0, pool_size=(2, 2), strides=(2, 2), padding="none")
        l1.forward()
        self.assertTrue(np.all(t1_ref_img == l1.output))

        # Backward
        l1.delta = np.ones_like(t1_ref_img)
        l1.backward()
        self.assertTrue(np.all(t1_ref_back == l0.delta))

    def test_avgpool_2x2_same(self):
        # Test 1
        t1_in_img = np.array([[
            [
                [0, 1, 0, 4, 5],
                [2, 3, 2, 1, 3],
                [4, 4, 0, 4, 3],
                [2, 5, 2, 6, 4],
                [1, 0, 0, 5, 7],
            ]
        ]])
        t1_ref_img = np.array([[
            [
                [1.5, 1.75, 2.0],
                [3.75, 3.0, 1.75],
                [0.25, 1.25, 1.75],
            ]
        ]])

        t1_ref_back = np.array([[
            [
                [0.25, 0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25, 0.25],
            ]
        ]])

        # Test 1
        # Forward
        l0 = Input(shape=t1_in_img[0].shape)
        l0.output = t1_in_img
        l1 = AvgPool(l0, pool_size=(2, 2), strides=(2, 2), padding="same")
        l1.forward()
        self.assertTrue(np.all(t1_ref_img == l1.output))

        # Backward
        l1.delta = np.ones_like(t1_ref_img)
        l1.backward()
        self.assertTrue(np.all(t1_ref_back == l0.delta))


if __name__ == "__main__":
    unittest.main()
