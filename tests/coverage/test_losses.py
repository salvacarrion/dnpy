import unittest

from dnpy import utils
from dnpy.losses import *
from dnpy.net import Net
from dnpy.optimizers import *
import torch
import torch.nn as nn
import numpy as np


class TestLossesMethods(unittest.TestCase):

    def test_binary_cross_entropy(self):
        # Test 1
        y_pred = np.array([
            [0.7],
            [0.3],
            [0.9],
            [0.1],
            [0.6],
        ])
        y_true = np.array([
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [1.0],
        ])

        delta_ref = np.array([
            [-1.42857122],
            [-3.33333222],
            [-1.11111099],
            [ 1.11111099],
            [-1.66666639]
        ])

        loss = BinaryCrossEntropy()

        # Test: Value
        value = loss.compute_loss(y_pred, y_true)
        self.assertAlmostEqual(value, 0.4564, places=4)

        # Test: Delta
        delta = loss.compute_delta(y_pred, y_true)
        self.assertTrue(np.allclose(delta_ref, delta, atol=1e-4))



if __name__ == "__main__":
    unittest.main()
