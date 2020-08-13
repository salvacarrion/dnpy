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


def fw_bw(model, x_train_mb, y_train_mb):
    # Forward
    model.feed_input(x_train_mb)
    model.forward()

    # Compute loss
    losses, deltas = model.compute_losses(y_target=y_train_mb)

    # Backward
    model.reset_grads()
    model.do_delta(deltas)
    model.backward()

    # Results
    return losses, deltas


def _gradient_check(model, x_train, y_train, max_samples, epsilon=10e-7, burn_in_passes=0):
    # Pass 1 ( x + 0 )
    for i in range(burn_in_passes+1):
        fw_bw(model, x_train, y_train)
        if burn_in_passes > 0:
            model.apply_grads()

    # Get params and grads
    params0 = model.get_params(only_trainable=True)
    grads0 = model.get_grads()

    # Transform params and grads to vector
    vparams, params_slides = utils.params2vector(params0)
    vgrads, grads_slides = utils.params2vector(grads0)

    sampled_idxs = utils.sample_params_slices(params_slides, max_samples=max_samples)
    sampled_idxs = np.array(sampled_idxs)

    # Reserve memory (reduced by sampling)
    analytical_grads = vgrads[sampled_idxs]
    numerical_grads = np.zeros(analytical_grads.shape)

    # Grad check
    for i, idx in enumerate(sampled_idxs):
        print(f"{i+1}/{len(sampled_idxs)}")
        # f(theta + epsilon)
        new_vector_plus = copy.deepcopy(vparams)
        new_vector_plus[idx] += epsilon
        new_params_plus = utils.vector2params(new_vector_plus, params0)

        # Set params and get loss
        model.set_params(new_params_plus, only_trainable=True)
        losses_plus, _ = fw_bw(model, x_train, y_train)

        # f(theta - epsilon)
        new_vector_minus = copy.deepcopy(vparams)
        new_vector_minus[idx] -= epsilon
        new_params_minus = utils.vector2params(new_vector_minus, params0)

        # Set params and get loss
        model.set_params(new_params_minus, only_trainable=True)
        losses_minus, _ = fw_bw(model, x_train, y_train)

        # Numerical dx
        numerical_dx_loss = (losses_plus[0] - losses_minus[0]) / (2.0 * epsilon)
        numerical_grads[i] = numerical_dx_loss  # i => index not sampled

        # For debugging
        # # Analytical
        # analytical_dx_loss = analytical_grads[i]  # i => index not sampled
        #
        # # Single error
        # numerator = abs(analytical_dx_loss - numerical_dx_loss)
        # denominator = max(analytical_dx_loss, numerical_dx_loss)
        # rel_err = float(numerator / denominator)
        # print(f"\t- Relative error (grad[{i}]): {rel_err}")
        asd = 3

    # Compute relative error
    numerator = np.linalg.norm(analytical_grads-numerical_grads)
    denominator = np.linalg.norm(analytical_grads) + np.linalg.norm(numerical_grads)
    rel_err = numerator/denominator
    return rel_err


def gradient_check(model, x_train, y_train, batch_size, max_error=10e-7, max_samples=5, verbose=True):
    # Setup
    num_samples = len(x_train[0])
    num_batches = int(num_samples / batch_size)
    if model.mode != "train":
        model.set_mode('train')

    # Train mini-batches
    for b in range(num_batches):
        # Get minibatch
        x_train_mb, y_train_mb = model.get_minibatch(x_train, y_train, b, batch_size)

        # Compute error
        rel_err = _gradient_check(model, x_train_mb, y_train_mb, max_samples)
        if verbose:
            print("Batch {}/{}: Relative error={}".format(b + 1, num_batches, rel_err))

        if rel_err > max_error:
            return False
    return True


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
        c = 1  # 0.8
        tr_size = int(len(X) * c)
        x_train, y_train = X[:tr_size], Y[:tr_size]

        # Define architecture
        l_in = Input(shape=x_train[0].shape)
        l = l_in
        l = Dense(l, 15)
        l = Tanh(l)
        l = Dense(l, 1)
        l_out = l

        # Build network
        model = Net()
        model.build(
            l_in=[l_in],
            l_out=[l_out],
            optimizer=Adam(lr=10e-2),
            losses=[losses.MSE()],
            metrics=[[metrics.MSE(), metrics.MAE()]],
            debug=False
        )

        # Check gradient
        passed = gradient_check(model, [x_train], [y_train], batch_size=int(len(x_train)/10), max_samples=25)
        self.assertTrue(passed)

    def test_iris_model1(self):
        from sklearn import datasets

        # Get data
        iris = datasets.load_iris()
        X = iris.data  # we only take the first two features.
        Y = iris.target

        # Pre-processing
        # Standarize
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Classes to categorical
        num_classes = 3
        Y = utils.to_categorical(Y, num_classes=num_classes)

        # Shuffle dataset
        idxs = np.arange(len(X))
        np.random.shuffle(idxs)
        X, Y = X[idxs], Y[idxs]

        # Select train/test
        c = 1  # 0.8
        tr_size = int(len(X) * c)
        x_train, y_train = X[:tr_size], Y[:tr_size]

        # Define architecture
        l_in = Input(shape=x_train[0].shape)
        l = Dense(l_in, 20)
        l = Relu(l)
        l = Dense(l, 15)
        l = Relu(l)
        l = Dense(l, num_classes)
        l_out = Softmax(l)

        # Build network
        model = Net()
        model.build(
            l_in=[l_in],
            l_out=[l_out],
            optimizer=Adam(lr=10e-2),
            losses=[losses.CrossEntropy()],
            metrics=[[metrics.CategoricalAccuracy()]],
            debug=False
        )

        # Check gradient
        passed = gradient_check(model, [x_train], [y_train], batch_size=int(len(x_train)/10), max_samples=25)
        self.assertTrue(passed)

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

        # Define architecture
        l_in = Input(shape=x_train[0].shape)
        l = Reshape(l_in, shape=(28 * 28,))
        l = Relu(Dense(l, 1024))
        # l = BatchNorm(l)
        l = LeakyRelu(Dense(l, 1024), alpha=0.1)
        l_out = Softmax(Dense(l, num_classes))

        # Build network
        model = Net()
        model.build(
            l_in=[l_in],
            l_out=[l_out],
            optimizer=Adam(lr=10e-3),
            losses=[losses.CrossEntropy()],
            metrics=[[metrics.CategoricalAccuracy()]],
            debug=False,
            smart_derivatives=False,
        )

        # Check gradient
        passed = gradient_check(model, [x_train], [y_train], batch_size=int(len(x_train)/10), max_samples=5)
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
