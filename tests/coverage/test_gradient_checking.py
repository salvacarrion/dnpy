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


def get_params(model):
    params = []
    for i, l in enumerate(model.fts_layers, 0):
        p = copy.deepcopy(l.params)
        params.append(p)
    return params


def set_params(model, params):
    for i, l in enumerate(model.fts_layers, 0):
        l.params = params[i]


def get_grads(model):
    grads = []
    for i, l in enumerate(model.fts_layers, 0):
        g = copy.deepcopy(l.grads)
        grads.append(g)
    return grads

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


def set_backward_delta_grads(model, backward_grads):
    for i, l in enumerate(model.fts_layers, 0):
        l.grads = backward_grads[i]
        

def test_fw_bw(model, x_train, y_train):
    x_train_mb, y_train_mb = [x_train], [y_train]
    model.feed_input(x_train_mb)
    model.forward()
    b_losses2, b_deltas2 = model.compute_losses(y_target=y_train_mb)
    model.reset_grads()
    model.do_delta(b_deltas2)
    model.backward()

    # Results
    backward_deltas2, backward_grads2 = get_backward_delta_grads(model)
    return (b_losses2, b_deltas2), (backward_deltas2, backward_grads2)

def params2vector(params):
    vector = []
    for li in range(len(params)):
        for kp, vp in params[li].items():
            vector.append(vp.reshape(-1, 1))
    vector = np.concatenate(vector, axis=0)
    return vector


def vector2params(vector, params):
    pi = 0
    new_params = copy.deepcopy(params)
    for li in range(len(params)):
        for kp, vp in params[li].items():
            new_vp = vector[pi:pi+vp.size]
            new_vp = np.reshape(new_vp, vp.shape)
            new_params[li][kp] = new_vp
            pi += vp.size
    return new_params


def check_gradient_model(model, x_train, y_train, epsilon=10e-7):
    # Train model
    model.set_mode('train')

    model.reset_grads()

    # Pass 1 ( x + 0 )
    (b_losses1, b_deltas1), (backward_deltas1, backward_grads1) = (None, None), (None, None)

    for i in range(100):
        (b_losses1, b_deltas1), (backward_deltas1, backward_grads1) = test_fw_bw(model, x_train, y_train)
        model.apply_grads()
        print(b_losses1[0])
    model_plus = copy.deepcopy(model)
    model_minus = copy.deepcopy(model)

    # Get params and grads
    params1 = get_params(model)
    grads1 = get_grads(model)

    # Transform params and grads to vector
    vparams = params2vector(params1)
    vgrads = params2vector(grads1)

    # Reserve memory
    analytical_grads = vgrads
    numerical_grads = np.zeros(vparams.shape)

    # Grad check
    for i in range(len(vparams)):
        # f(theta + epsilon)
        new_vector_plus = copy.deepcopy(vparams)
        new_vector_plus[i] += epsilon
        new_params_plus = vector2params(new_vector_plus, params1)

        set_params(model_plus, new_params_plus)
        (b_losses_plus, b_deltas_plus), (backward_deltas_plus, backward_grads_plus) = test_fw_bw(model_plus, x_train, y_train)

        # f(theta - epsilon)
        new_vector_minus = copy.deepcopy(vparams)
        new_vector_minus[i] -= epsilon
        new_params_minus = vector2params(new_vector_minus, params1)

        set_params(model_minus, new_params_minus)
        (b_losses_minus, b_deltas_minus), (backward_deltas_minus, backward_grads_minus) = test_fw_bw(model_minus, x_train, y_train)

        # Numerical dx
        numerical_J_plus_loss = b_losses_plus[0]
        numerical_J_minus_loss = b_losses_minus[0]
        numerical_dx_loss = (numerical_J_plus_loss - numerical_J_minus_loss) / (2.0 * epsilon)
        numerical_grads[i] = numerical_dx_loss

        # Analytical
        analytical_dx_loss = analytical_grads[i]

        # Single error
        numerator = abs(analytical_dx_loss - numerical_dx_loss)
        denominator = max(analytical_dx_loss, numerical_dx_loss)
        rel_err = numerator / denominator
        asdasd = 3

    # Compute relative error
    numerator = np.linalg.norm(analytical_grads-numerical_grads)
    denominator = np.linalg.norm(analytical_grads) + np.linalg.norm(numerical_grads)
    rel_err = numerator/denominator

    if rel_err < 10e-7:
        print("ok")
    else:
        print("not ok")

    return 2


class TestGradCheck(unittest.TestCase):

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
        Y = (Y == 1).astype(float).reshape((-1, 1))

        # Shuffle dataset
        idxs = np.arange(len(X))
        np.random.shuffle(idxs)
        X, Y = X[idxs], Y[idxs]

        # Select train/test
        c = 1#0.8
        tr_size = int(len(X) * c)
        x_train, y_train = X[:tr_size], Y[:tr_size]

        # Params *********************************
        batch_size = int(len(x_train) / 5)
        epochs = 1000

        # Define architecture
        l_in = Input(shape=x_train[0].shape)
        l = Dense(l_in, 20)
        l = Relu(l)
        l = Dense(l, 1)
        l_out = Sigmoid(l)

        # Build network
        model = Net()
        model.build(
            l_in=[l_in],
            l_out=[l_out],
            optimizer=Adam(lr=0.1),
            losses=[losses.BinaryCrossEntropy()],
            metrics=[[metrics.BinaryAccuracy()]],
            debug=False
        )

        rel_err = check_gradient_model(model, x_train, y_train, epsilon=10e-5)
        asdasd = 3

    def test_boston_model1(self):
        from sklearn import datasets

        # Get data
        X, Y = datasets.load_boston(return_X_y=True)
        Y = Y.reshape((-1, 1))

        # Pre-processing
        # Normalize
        # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # # Shuffle dataset
        # idxs = np.arange(len(X))
        # np.random.shuffle(idxs)
        # X, Y = X[idxs], Y[idxs]

        # Select train/test
        c = 1#0.8
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
            optimizer=Adam(lr=10e-3),
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
        # x_train, y_train = utils.shuffle_dataset(x_train, y_train)
        # x_train = x_train[:100]
        # y_train = y_train[:100]

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
