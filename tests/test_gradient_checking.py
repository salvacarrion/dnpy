import unittest
import copy
import math

from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy.regularizers import *
from dnpy import metrics, losses
from dnpy import utils

np.random.seed(42)


def get_backward_delta_grads(model):
    backward_grads = []
    backward_deltas = []
    for i, l in enumerate(model.fts_layers, 1):
        # Grads
        grads = copy.deepcopy(l.grads)
        backward_grads.append(grads)

        # Deltas
        deltas = copy.deepcopy(l.delta)
        backward_deltas.append(deltas)
    return backward_deltas, backward_grads


def check_gradient_model(model, x_train, y_train, epsilon=10e-5):
    # Train model
    model.set_mode('train')

    # Pass 1 ( x + 0 )
    x_train_mb, y_train_mb = [x_train], [y_train]
    model.reset_grads()
    model.feed_input(x_train_mb)
    model.forward()
    b_losses1, b_deltas1 = model.compute_losses(y_target=y_train_mb)
    model.do_delta(b_deltas1)
    model.backward()

    # Results
    backward_deltas1, backward_grads1 = get_backward_delta_grads(model)

    # Pass 2 ( x + epsilon )
    x_train_mb, y_train_mb = [x_train + epsilon], [y_train]
    model.reset_grads()
    model.feed_input(x_train_mb)
    model.forward()
    b_losses2, b_deltas2 = model.compute_losses(y_target=y_train_mb)
    model.do_delta(b_deltas2)
    model.backward()

    # Results
    backward_deltas2, backward_grads2 = get_backward_delta_grads(model)

    # Pass 3 ( x - epsilon )
    x_train_mb, y_train_mb = [x_train - epsilon], [y_train]
    model.reset_grads()
    model.feed_input(x_train_mb)
    model.forward()
    b_losses3, b_deltas3 = model.compute_losses(y_target=y_train_mb)
    model.do_delta(b_deltas3)
    model.backward()

    # Results
    backward_deltas3, backward_grads3 = get_backward_delta_grads(model)

    # Gradient
    analytical_dx = np.mean(np.sum(b_deltas1[0], axis=1), axis=0)
    numerical_dx = (b_losses2[0] - b_losses3[0]) / (2.0 * epsilon)

    # Check gradient
    abs_err = math.fabs(analytical_dx - numerical_dx)
    max_err = max(analytical_dx, numerical_dx)
    rel_err = abs_err / max_err

    if rel_err < 10e-4:
        print("ok")
    else:
        print("not ok")

    return rel_err


class TestGradCheck(unittest.TestCase):

    def test_boston_model1(self):
        from sklearn import datasets

        # Get data
        X, Y = datasets.load_boston(return_X_y=True)
        Y = Y.reshape((-1, 1))

        # Pre-processing
        # Normalize
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Shuffle dataset
        idxs = np.arange(len(X))
        np.random.shuffle(idxs)
        X, Y = X[idxs], Y[idxs]

        # Select train/test
        c = 0.8
        tr_size = int(len(X) * c)
        x_train, y_train = X[:tr_size], Y[:tr_size]

        # Define architecture
        l_in = Input(shape=x_train[0].shape)
        l = l_in
        l = Dense(l, 20)
        l = Relu(l)
        l = Dense(l, 15)
        l = Relu(l)
        l = Dense(l, 1)
        l_out = l

        # Build network
        model = Net()
        model.build(
            l_in=[l_in],
            l_out=[l_out],
            optimizer=RMSProp(lr=0.01),
            losses=[losses.MSE()],
            metrics=[[metrics.MSE(), metrics.MAE()]],
            debug=False
        )

        rel_err = check_gradient_model(model, x_train, y_train, epsilon=10e-5)
        asdasd = 3

    def test_mnist_model1(self):
        from keras import datasets

        # Get data
        (x_train, y_train), (_, _) = datasets.mnist.load_data()

        # Pre-processing
        # Normalize
        x_train = x_train / 255.0

        # Classes to categorical
        num_classes = 10
        y_train = utils.to_categorical(y_train, num_classes=num_classes)

        # Shuffle dataset
        x_train, y_train = utils.shuffle_dataset(x_train, y_train)
        x_train = x_train[:100]
        y_train = y_train[:100]

        # Define architecture
        l_in = Input(shape=x_train[0].shape)
        l = Reshape(l_in, shape=(28 * 28,))
        l = PRelu(Dense(l, 1024))
        l_out = Softmax(Dense(l, num_classes))

        # Build network
        model = Net()
        model.build(
            l_in=[l_in],
            l_out=[l_out],
            optimizer=Adam(lr=0.001),
            losses=[losses.CrossEntropy()],
            metrics=[[metrics.CategoricalAccuracy()]],
            debug=False,
            smart_derivatives=False,
        )

        rel_err = check_gradient_model(model, x_train, y_train, epsilon=10e-5)


if __name__ == "__main__":
    unittest.main()
