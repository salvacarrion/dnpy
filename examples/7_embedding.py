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


def main():
    # define documents
    docs = ['Well done!',
            'Good work',
            'Great effort',
            'nice work',
            'Excellent!',
            'Weak',
            'Poor effort!',
            'not good',
            'poor work',
            'Could have done better.']
    # define class labels
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    num_classes = 1  # Binary

    # integer encode the documents
    vocab_size = 20
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    print(encoded_docs)

    # pad documents to a max length of 4 words
    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)

    # Dataset
    x_train = padded_docs
    y_train = np.expand_dims(labels, axis=1)

    # Params *********************************
    batch_size = int(len(x_train)/1)
    epochs = 1000

    # Define architecture
    l_in = Input(shape=x_train[0].shape)  # Technically not needed
    l = l_in
    l = Embedding(l, input_dim=vocab_size, output_dim=8, input_length=max_length)
    l = Reshape(l, shape=(-1))
    l_out = Sigmoid(Dense(l, num_classes))

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

    # Train
    mymodel.fit([x_train], [y_train],
                x_test=None, y_test=None,
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=False,
                print_rate=1)

    # Evaluate
    # m = mymodel.evaluate([x_test], [y_test], batch_size=batch_size)


if __name__ == "__main__":
    main()
