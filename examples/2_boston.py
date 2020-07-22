from sklearn.datasets import load_boston

from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy import metrics, losses

# For debugging
# np.random.seed(123)

# import keras


def mlp_regression():
    # Load data  ******************************
    X, Y = load_boston(return_X_y=True)
    Y = np.reshape(Y, (len(Y), 1))

    # Shuffle data
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X, Y = X[idxs], Y[idxs]

    # Split data
    c = 0.9
    ds_size = int(len(X) * c)
    x_train, y_train = X[:ds_size], Y[:ds_size]
    x_test, y_test = X[ds_size:], Y[ds_size:]

    # Params *********************************
    batch_size = len(x_train)
    epochs = 10

    # Define architecture
    l_in = Input(shape=(len(X[0]),))
    l = l_in
    l = Dense(l, 15)
    l = Relu(l)
    l = Dense(l, 1)
    l = Relu(l)
    l_out = l

    # Build network
    mymodel = Net()
    mymodel.build(
        l_in=[l_in],
        l_out=[l_out],
        opt=SGD(lr=0.01),
        losses=[losses.MSE()],
        metrics=[metrics.MSE()],
    )

    # Print model
    mymodel.summary(batch_size=batch_size)

    # Train
    mymodel.fit(x_train, y_train,
                x_test=x_test, y_test=y_test,
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=False)

    # # Evaluate
    # m = mymodel.evaluate(x_test, y_test, batch_size=batch_size)


if __name__ == "__main__":
    mlp_regression()
