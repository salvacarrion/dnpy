from sklearn.datasets import load_boston

from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy import metrics, losses

# For debugging
np.random.seed(42)


def main():
    X = np.array([
        [0.1, 0.2, 0.7],
        [0.1, 0.2, 0.7],
        [0.1, 0.2, 0.7],
        [0.1, 0.2, 0.7],
        [0.1, 0.2, 0.7],
    ])
    Y = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
                  ])

    x_train, y_train = X, Y
    x_test, y_test = X, Y

    # Params *********************************
    batch_size = len(x_train)
    epochs = 5

    # Define architecture
    l_in = Input(shape=(len(X[0]),))
    l = l_in
    l = Dense(l, 3)
    l = Relu(l)
    l = Dense(l, 3)
    l = Sigmoid(l)
    l = Dense(l, 3)
    l = Softmax(l)
    l_out = l

    # Build network
    mymodel = Net()
    mymodel.build(
        l_in=[l_in],
        l_out=[l_out],
        opt=SGD(lr=0.1),
        losses=[losses.BinaryCrossEntropy()],
        metrics=[metrics.CategoricalAccuracy()],
        debug=False
    )

    # Print model
    #mymodel.summary(batch_size=batch_size)

    # Train
    mymodel.fit(x_train, y_train,
                x_test=x_test, y_test=y_test,
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=True)

    # # Evaluate
    # m = mymodel.evaluate(x_test, y_test, batch_size=batch_size)


if __name__ == "__main__":
    main()
