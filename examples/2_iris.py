from sklearn import datasets

from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy.regularizers import *
from dnpy import metrics, losses
from dnpy import utils

# For debugging
np.random.seed(42)


def main():
    # Get data
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    Y = iris.target

    # Pre-processing
    # Normalize
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

    # Classes to categorical
    num_classes = 3
    Y = utils.to_categorical(Y, num_classes=num_classes)

    # Shuffle dataset
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X, Y = X[idxs], Y[idxs]

    # Select train/test
    c = 0.8
    tr_size = int(len(X) * c)
    x_train, y_train = X[:tr_size], Y[:tr_size]
    x_test, y_test = X[tr_size:], Y[tr_size:]

    # Params *********************************
    batch_size = int(len(x_train)/5)
    epochs = 1000

    # Define architecture
    l_in = Input(shape=x_train[0].shape)
    l = Dense(l_in, 20, kernel_regularizer=L2(lmda=0.01), bias_regularizer=L1(lmda=0.01))
    l = Relu(l)
    l = Dense(l, 15)
    l = BatchNorm(l)
    l = Dropout(l, 0.1)
    l = Relu(l)
    l = Dense(l, num_classes)
    l_out = Softmax(l)

    # Build network
    mymodel = Net()
    mymodel.build(
        l_in=[l_in],
        l_out=[l_out],
        optimizer=Adam(lr=0.1),
        losses=[losses.CrossEntropy()],
        metrics=[metrics.CategoricalAccuracy()],
        debug=False
    )

    # Print model
    mymodel.summary(batch_size=batch_size)

    # Train
    mymodel.fit([x_train], [y_train],
                x_test=[x_test], y_test=[y_test],
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=True,
                print_rate=10)

    # Evaluate
    # m = mymodel.evaluate([x_test], [y_test], batch_size=batch_size)


if __name__ == "__main__":
    main()
