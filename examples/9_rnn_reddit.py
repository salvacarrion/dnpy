import keras
from keras import layers
from keras import datasets
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import csv
import numpy as np
import itertools
import nltk

import numpy as np
from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy.regularizers import *
from dnpy import metrics, losses
from dnpy import utils

# For debugging
np.random.seed(42)
nltk.download('punkt')

def vectorize(samples, length, dimension):
    results = np.zeros((len(samples), length, dimension))
    for i, words_idxs in enumerate(samples):
        results[i, words_idxs] = 1
    return results

def getSentenceData(path, vocabulary_size=8000):
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # Filter the sentences having few words (including SENTENCE_START and SENTENCE_END)
    tokenized_sentences = list(filter(lambda x: len(x) > 3, tokenized_sentences))

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print("\nExample sentence: '%s'" % sentences[1])
    print("\nExample sentence after Pre-processing: '%s'\n" % tokenized_sentences[0])

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))

    # Print an training data example
    x_example, y_example = X_train[17], y_train[17]
    print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
    print("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))

    return X_train, y_train, index_to_word, word_to_index


def save_data(filename):
    word_dim = 8000
    X_train, y_train, index_to_word, word_to_index = getSentenceData('./data/reddit/reddit-comments-2015-08.csv',
                                                                     word_dim)
    data = {"X_train": X_train, "y_train": y_train,
            "index_to_word": index_to_word, "word_to_index": word_to_index}

    # Save data
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
    print("Data saved!")


def load_data(filename):
    # Load data
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
        X_train = data["X_train"]
        y_train = data["y_train"]
        index_to_word = data["index_to_word"]
        word_to_index = data["word_to_index"]
    return X_train, y_train, index_to_word, word_to_index


def main():
    num_classes = 8000
    hidden_dim = 100
    # save_data("./data/reddit/sent.pkl")
    x_train, y_train, index_to_word, word_to_index = load_data("./data/reddit/sent.pkl")

    # To categorical (one-hot)
    x_tmp, y_tmp = [], []
    for i in range(len(x_train)):
        x_t = to_categorical(x_train[i], num_classes=num_classes)
        y_t = to_categorical(y_train[i], num_classes=num_classes)
        x_tmp.append(x_t)
        y_tmp.append(y_t)
    x_train = x_tmp
    y_train = y_tmp

    batch_size = 1#int(len(x_train) / 8)
    epochs = 100

    # Define architecture
    l_in = Input(shape=(None, num_classes))
    l = l_in
    # l = Embedding(l, input_dim=max_words, output_dim=8, input_length=max_length)
    l = SimpleRNN(l, hidden_dim=100, stateful=False, return_sequences=True, unroll=False, bptt_truncate=4)
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
                x_test=None, y_test=None,
                batch_size=batch_size, epochs=epochs,
                evaluate_epoch=False,
                print_rate=1)

    asdasd = 33


if __name__ == "__main__":
    main()
