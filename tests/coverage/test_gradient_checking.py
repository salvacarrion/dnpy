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


def _gradient_check(model, x_train, y_train, max_samples, epsilon, verbose=False):
    # Pass 1 ( x + 0 )
    fw_bw(model, x_train, y_train)

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
        if int(verbose) > 1:
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


def gradient_check(model, x_train, y_train, batch_size, max_error=10e-7, max_samples=5, burn_in_passes=0,
                   epsilon=10e-7, verbose=True):
    # Setup
    num_samples = len(x_train[0])
    num_batches = int(num_samples / batch_size)
    if model.mode != "train":
        model.set_mode('train')

    # Burn-in passes
    for i in range(burn_in_passes):
        print(f"- Burn-in pass... ({i+1}/{burn_in_passes})")
        for b in range(num_batches):
            # Get minibatch
            x_train_mb, y_train_mb = model.get_minibatch(x_train, y_train, b, batch_size)

            losses, _ = fw_bw(model, x_train_mb, y_train_mb)
            model.apply_grads()
            print(f"\t- (Batch {b+1}/{num_batches}) Burn-in losses: {losses}")

    # Grad check
    for b in range(num_batches):
        # Get minibatch
        x_train_mb, y_train_mb = model.get_minibatch(x_train, y_train, b, batch_size)

        # Compute error
        rel_err = _gradient_check(model, x_train_mb, y_train_mb, max_samples=max_samples, epsilon=epsilon, verbose=verbose)
        if verbose:
            print("Batch {}/{}: Relative error={}".format(b + 1, num_batches, rel_err))

        # Check relative error
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

    def test_mnist_model_mlp(self):
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

        l_in = Input(shape=x_train[0].shape)
        l = Reshape(l_in, shape=(28 * 28,))
        l1 = Relu(Dense(l, 512))
        # l2 = Relu(Dense(l1, 512))
        # l = Add([l1, l2])
        # l = BatchNorm(l)
        l = Relu(Dense(l1, 512))
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
        passed = gradient_check(model, [x_train], [y_train], batch_size=int(len(x_train)/10), max_samples=5, verbose=2)
        self.assertTrue(passed)

    def test_mnist_model_conv(self):
        from keras import datasets

        # Get data
        (x_train, y_train), (_, _) = datasets.mnist.load_data()

        # Pre-processing

        # Add channel dimension
        x_train = np.expand_dims(x_train, axis=1)

        # Normalize
        x_train = x_train / 255.0

        # Classes to categorical
        num_classes = 10
        y_train = utils.to_categorical(y_train, num_classes=num_classes)

        # Shuffle dataset
        x_train, y_train = utils.shuffle_dataset(x_train, y_train)

        # Define architecture
        l_in = Input(shape=x_train[0].shape)
        l = l_in

        l = Conv2D(l, filters=2, kernel_size=(3, 3), strides=(1, 1), padding="none")
        l = MaxPool2D(l, pool_size=(3, 3), strides=(2, 2), padding="none")
        l = Relu(l)
        # l = GaussianNoise(l, stddev=0.1)

        # l = Conv2D(l, filters=4, kernel_size=(3, 3), strides=(1, 1), padding="same")
        # l = MaxPool2D(l, pool_size=(3, 3), strides=(2, 2), padding="none")
        # l = Relu(l)

        # l = DepthwiseConv2D(l, kernel_size=(3, 3), strides=(1, 1), padding="none")
        # l = PointwiseConv2D(l, filters=1)
        # l = MaxPool2D(l, pool_size=(3, 3), strides=(2, 2), padding="none")
        # l = Relu(l)

        l = Reshape(l, shape=(-1))
        l = Dense(l, num_classes, kernel_initializer=initializers.RandomUniform())
        l_out = Softmax(l)

        # Build network
        model = Net()
        model.build(
            l_in=[l_in],
            l_out=[l_out],
            optimizer=Adam(lr=10e-2),
            losses=[losses.CrossEntropy()],
            metrics=[[metrics.CategoricalAccuracy()]],
            debug=False,
            smart_derivatives=False,
        )

        # Check gradient
        passed = gradient_check(model, [x_train], [y_train], batch_size=int(len(x_train)/10), max_samples=5, verbose=2)
        self.assertTrue(passed)

    def test_embedding_model1(self):
        from keras.preprocessing.text import one_hot
        from keras.preprocessing.sequence import pad_sequences

        # define documents
        docs = ['Well done!',
          'Good work',
          'Great effort',
          'nice work',
          'Excellent!',
          'Weak',
          'Poor effort!',
          'not good',
          'poor work',
          'Could have done better.']
        # define class labels
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        num_classes = 1  # Binary

        # integer encode the documents
        vocab_size = 20
        encoded_docs = [one_hot(d, vocab_size) for d in docs]

        # pad documents to a max length of 4 words
        max_length = 4
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

        # Dataset
        x_train = padded_docs
        y_train = np.expand_dims(labels, axis=1)

        # Define architecture
        l_in = Input(shape=x_train[0].shape)  # Technically not needed
        l = l_in
        l = Embedding(l, input_dim=vocab_size, output_dim=8, input_length=max_length)
        l = Reshape(l, shape=(-1))
        l_out = Sigmoid(Dense(l, num_classes))

        # Build network
        model = Net()
        model.build(
            l_in=[l_in],
            l_out=[l_out],
            optimizer=Adam(lr=10e-2),
            losses=[losses.BinaryCrossEntropy()],
            metrics=[[metrics.BinaryAccuracy()]],
            debug=False,
            smart_derivatives=True,
        )

        # Check gradient
        passed = gradient_check(model, [x_train], [y_train], batch_size=int(len(x_train)/1), max_samples=5,
                                burn_in_passes=0)
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
