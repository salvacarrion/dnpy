from keras import datasets
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

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

    # x_train = np.expand_dims(x_train, axis=2)
    # x_test = np.expand_dims(x_test, axis=2)
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    # Params *********************************
    batch_size = int(len(x_train) / 10)
    epochs = 10

    # Define architecture
    l_in = Input(shape=x_train[0].shape)
    l = l_in
    l = Embedding(l, input_dim=max_words, output_dim=8, input_length=max_length)
    l = SimpleRNN(l, units=32, stateful=False, return_sequences=False, unroll=False)
    l = Dense(l, units=1)
    l_out = Sigmoid(l)

    # Build network
    mymodel = Net()
    mymodel.build(
        l_in=[l_in],
        l_out=[l_out],
        optimizer=Adam(lr=0.1),
        losses=[losses.BinaryCrossEntropy()],
        metrics=[[metrics.BinaryAccuracy()]],
        debug=False,
        smart_derivatives=True,
    )

    # Print model
    mymodel.summary()

    # Train
    mymodel.fit([x_train], [y_train],
                x_test=None, y_test=None,
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=False,
                print_rate=1)

    sdfsd = 33

    asdasd = 33


if __name__ == "__main__":
    main()
