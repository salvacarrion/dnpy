import numpy as np
from matplotlib import pyplot as plt


def to_categorical(y, num_classes):
    tmp = np.zeros((len(y), num_classes))
    tmp[np.arange(len(y)), y] = 1.0
    return tmp


def to_binary(y, label):
    tmp = np.zeros_like(y)
    tmp[y == label] = 1.0
    return tmp


def to_label(y):
    return np.argmax(y, axis=1)


def shuffle_dataset(x, y):
    idxs = np.arange(len(x))
    np.random.shuffle(idxs)
    return x[idxs], y[idxs]


def split_dataset(x, y, split=0.8):
    size = int(len(x)*split)
    return x[:size], y[:size]


def show_dataset(x, y, size=1, show_rnd=True):
    if show_rnd:
        indices = np.random.randint(0, len(x), size=size)
    else:
        indices = np.arange(0, size)

    for i, idx in enumerate(indices, 1):
        print(f"[{i}/{size}] Index: {idx}\t Y: {y[idx]}")
        plt.imshow(x[idx])
        plt.show()
