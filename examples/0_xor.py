from sklearn.datasets import load_boston

from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy import metrics, losses

# For debugging
np.random.seed(1)


def main():
    # Get data
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    Y = np.array([[-1],
                  [1],
                  [1],
                  [-1]])

    x_train, y_train = X, Y
    x_test, y_test = X, Y

    # Params *********************************
    batch_size = len(x_train)
    epochs = 60000

    # Define architecture
    l_in = Input(shape=x_train[0].shape)
    l = l_in
    l = Dense(l, 4)
    l = Tanh(l)
    l = Dense(l, 1)
    l_out = l

    # Build network
    mymodel = Net()
    mymodel.build(
        l_in=[l_in],
        l_out=[l_out],
        optimizer=SGD(lr=1.0, momentum=0.9, nesterov=False),
        losses=[losses.Hinge()],
        metrics=[[metrics.BinaryAccuracy(threshold=0.0), metrics.MAE()]],
        debug=False
    )

    # Print model
    mymodel.summary()

    # Train
    mymodel.fit([x_train], [y_train],
                x_test=[x_test], y_test=[y_test],
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=True,
                print_rate=100)

    # Evaluate
    m = mymodel.evaluate([x_test], [y_test], batch_size=batch_size)


if __name__ == "__main__":
    main()
