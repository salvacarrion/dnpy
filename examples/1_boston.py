from sklearn import datasets

from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy import metrics, losses

# For debugging
np.random.seed(42)


def main():
    X, Y = datasets.load_boston(return_X_y=True)
    Y = Y.reshape((-1, 1))

    # Preprocessing
    # Normalize
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

    x_train, y_train = X, Y
    x_test, y_test = X, Y

    # Params *********************************
    batch_size = len(x_train)
    epochs = 1000

    # Define architecture
    l_in = Input(shape=(len(X[0]),))
    l = l_in
    l = Dense(l, 20)
    l = Relu(l)
    l = Dense(l, 15)
    l = Relu(l)
    l = Dense(l, 1)
    l_out = l

    # Build network
    mymodel = Net()
    mymodel.build(
        l_in=[l_in],
        l_out=[l_out],
        opt=SGD(lr=0.01),
        losses=[losses.MSE()],
        metrics=[metrics.MSE()],
        debug=False
    )

    # Print model
    # mymodel.summary(batch_size=100)

    # Train
    mymodel.fit(x_train, y_train,
                x_test=x_test, y_test=y_test,
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=True,
                print_rate=10)

    # # Evaluate
    # m = mymodel.evaluate(x_test, y_test, batch_size=batch_size)


if __name__ == "__main__":
    main()
