from sklearn import datasets

from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy import metrics, losses

# For debugging
np.random.seed(42)


def main():
    # Get data
    X, Y = datasets.load_boston(return_X_y=True)
    Y = Y.reshape((-1, 1))

    # Pre-processing
    # Normalize
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

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
    batch_size = len(x_train)
    epochs = 1000

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
    mymodel = Net()
    mymodel.build(
        l_in=[l_in],
        l_out=[l_out],
        optimizer=RMSProp(lr=0.01),
        losses=[losses.MSE()],
        metrics=[[metrics.MSE(), metrics.MAE()]],
        debug=False
    )

    # Print model
    mymodel.summary()

    # Train
    mymodel.fit([x_train], [y_train],
                x_test=[x_test], y_test=[y_test],
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=True,
                print_rate=10)

    # # Evaluate
    # m = mymodel.evaluate([x_test], [y_test], batch_size=batch_size)


if __name__ == "__main__":
    main()
