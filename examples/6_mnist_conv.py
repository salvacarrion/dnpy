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
    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    # Normalize
    x_train = x_train/255.0
    x_test = x_test/255.0

    # Classes to categorical
    num_classes = 10
    y_train = utils.to_categorical(y_train, num_classes=num_classes)
    y_test = utils.to_categorical(y_test, num_classes=num_classes)

    # Shuffle dataset
    x_train, y_train = utils.shuffle_dataset(x_train, y_train)
    x_test, y_test = utils.shuffle_dataset(x_test, y_test)

    # Params *********************************
    batch_size = int(len(x_train)/1)
    epochs = 10

    # Define architecture
    l_in = Input(shape=x_train[0].shape)
    l = l_in

    # l = Conv2D(l, filters=1, kernel_size=(3, 3), strides=(1, 1), padding="none")
    # l = MaxPool(l, pool_size=(3, 3), strides=(2, 2), padding="none")
    # l = Relu(l)

    # l = Conv2D(l, filters=4, kernel_size=(3, 3), strides=(1, 1), padding="same")
    # l = MaxPool(l, pool_size=(3, 3), strides=(2, 2), padding="none")
    # l = Relu(l)

    l = DepthwiseConv2D(l, kernel_size=(3, 3), strides=(1, 1), padding="none")
    l = PointwiseConv2D(l, filters=1)
    l = MaxPool(l, pool_size=(3, 3), strides=(2, 2), padding="none")
    l = Relu(l)

    l = Reshape(l, shape=(-1))
    l = Dense(l, num_classes, kernel_initializer=initializers.RandomUniform())
    l_out = Softmax(l)

    # Build network
    mymodel = Net()
    mymodel.build(
        l_in=[l_in],
        l_out=[l_out],
        optimizer=Adam(lr=0.01),
        losses=[losses.CrossEntropy()],
        metrics=[[metrics.CategoricalAccuracy()]],
        debug=False,
        smart_derivatives=True,
    )

    # Print model
    mymodel.summary()

    # Train
    mymodel.fit([x_train], [y_train],
                x_test=[x_test], y_test=[y_test],
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=False,
                print_rate=1)

    # Evaluate
    # m = mymodel.evaluate([x_test], [y_test], batch_size=batch_size)


if __name__ == "__main__":
    main()
