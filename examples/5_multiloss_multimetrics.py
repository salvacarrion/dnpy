from keras import datasets


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
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # Pre-processing
    # Normalize
    x_train = x_train/255.0
    x_test = x_test/255.0

    # Classes to categorical
    num_classes = 10

    y_train_label = y_train.reshape((-1, 1))
    y_test_label = y_test.reshape((-1, 1))
    y_train = utils.to_categorical(y_train, num_classes=num_classes)
    y_test = utils.to_categorical(y_test, num_classes=num_classes)

    # Shuffle dataset
    x_train, y_train = utils.shuffle_dataset(x_train, y_train)
    x_test, y_test = utils.shuffle_dataset(x_test, y_test)

    # Params *********************************
    batch_size = int(len(x_train)/10)
    epochs = 10

    # Define architecture
    l_in1 = Input(shape=x_train[0].shape)
    l_in2 = Input(shape=x_train[0].shape)
    l_in3 = Input(shape=x_train[0].shape)

    l1 = Reshape(l_in1, shape=(28*28,))
    l2 = Reshape(l_in2, shape=(28*28,))
    l3 = Reshape(l_in2, shape=(28*28,))

    l1 = Tanh(Dense(l1, 300))
    l2 = Relu(Dense(l2, 300))
    l3 = Sigmoid(Dense(l3, 300))

    l_mid1 = Add([l1, l2])
    l_mid2 = Add([l1, l3])

    l1 = Relu(Dense(l_mid1, 300))
    l2 = Relu(Dense(l_mid2, 300))

    l2 = Add([l1, l2])
    l3 = Add([l2, l3])
    l1 = BatchNorm(l1)

    l_out1 = Softmax(Dense(l1, num_classes))
    l_out2 = Softmax(Dense(l2, num_classes))
    l_out3 = Relu(Dense(l3, 1))

    # Build network
    mymodel = Net()
    mymodel.build(
        l_in=[l_in1, l_in2, l_in3],
        l_out=[l_out1, l_out2, l_out3],
        optimizer=Adam(lr=0.001),
        losses=[losses.CrossEntropy(), losses.CrossEntropy(), losses.MSE()],
        metrics=[[metrics.CategoricalAccuracy()], [metrics.CategoricalAccuracy()], [metrics.MSE(), metrics.MAE()]],
        debug=False,
        smart_derivatives=True,
    )

    # Print model
    mymodel.summary()

    # Train
    mymodel.fit([x_train, x_train, x_train], [y_train, y_train, y_train_label],
                x_test=[x_test, x_test, x_test], y_test=[y_test, y_test, y_test_label],
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=False,
                print_rate=1)

    # Evaluate
    # m = mymodel.evaluate([x_test], [y_test], batch_size=batch_size)


if __name__ == "__main__":
    main()
