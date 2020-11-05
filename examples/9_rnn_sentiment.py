import keras
from keras import layers
from keras import datasets
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import numpy as np
from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy.regularizers import *
from dnpy import metrics, losses
from dnpy import utils

# For debugging
np.random.seed(42)


def vectorize(samples, length, dimension):
    results = np.zeros((len(samples), length, dimension))
    for i, words_idxs in enumerate(samples):
        results[i, words_idxs] = 1
    return results


def main():
    max_size = 500
    max_length = 150
    max_words = 1000

    # Get dataset
    (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(maxlen=max_length, num_words=max_words)
    x_train, y_train, x_test, y_test = x_train[:max_size], y_train[:max_size], x_test[:max_size], y_test[:max_size]

    # Pad sequences
    x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
    x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

    # To categorical (one-hot)
    x_tmp, y_tmp = [], []
    for i in range(len(x_train)):
        x_t = to_categorical(x_train[i], num_classes=max_words)
        x_tmp.append(x_t)
    x_train = np.stack(x_tmp, axis=0)

    # x_train = np.expand_dims(x_train, axis=2)
    # x_test = np.expand_dims(x_test, axis=2)
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    # Params *********************************
    batch_size = int(len(x_train) / 8)
    epochs = 30

    # inputs = keras.Input(shape=(max_length, max_words))
    # # x = layers.Embedding(input_dim=max_words, output_dim=8)(inputs)
    # x = layers.SimpleRNN(32, unroll=True)(inputs)
    # # Add a classifier
    # # x = layers.Flatten()(x)
    # # x = layers.Dense(64, activation="relu")(x)
    # outputs = layers.Dense(1, activation="sigmoid")(x)
    # model = keras.Model(inputs, outputs)
    # model.summary()
    #
    # model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    # # asdasd = 3
    #
    #

    # Define architecture
    l_in = Input(shape=(max_length, max_words))
    l = l_in
    # l = Reshape(l, -1)
    # l = Dense(l, 32)
    # l = Embedding(l, input_dim=max_words, output_dim=8, input_length=max_length)
    l_rnn = SimpleRNN(l, hidden_dim=32, stateful=False, return_sequences=False, unroll=False, bptt_truncate=4)
    l = l_rnn
    l_dense = Dense(l, units=1)
    l_out = Sigmoid(l_dense)

    # Build network
    mymodel = Net()
    mymodel.build(
        l_in=[l_in],
        l_out=[l_out],
        optimizer=Adam(lr=0.01),
        losses=[losses.BinaryCrossEntropy()],
        metrics=[[metrics.BinaryAccuracy()]],
        debug=False,
        smart_derivatives=True,
    )

    # Print model
    mymodel.summary()

    wba = np.load("/Users/salvacarrion/Documents/Programming/Python/dnpy/examples/data/simple_rnn-simple_rnn_cell-bias0.npy")
    wax = np.load("/Users/salvacarrion/Documents/Programming/Python/dnpy/examples/data/simple_rnn-simple_rnn_cell-kernel0.npy")
    waa = np.load("/Users/salvacarrion/Documents/Programming/Python/dnpy/examples/data/simple_rnn-simple_rnn_cell-recurrent_kernel0.npy")

    l_rnn.params['wba'] = np.expand_dims(wba, axis=0)
    l_rnn.params['wax'] = wax.T
    l_rnn.params['waa'] = waa.T

    b1 = np.load("/Users/salvacarrion/Documents/Programming/Python/dnpy/examples/data/dense-bias0.npy")
    w1 = np.load("/Users/salvacarrion/Documents/Programming/Python/dnpy/examples/data/dense-kernel0.npy")

    l_dense.params['b1'] = np.expand_dims(b1, axis=0)
    l_dense.params['w1'] = w1

    # Train
    mymodel.fit([x_train], [y_train],
                x_test=None, y_test=None,
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=True,
                print_rate=1)

    asdasd = 33


if __name__ == "__main__":
    main()
