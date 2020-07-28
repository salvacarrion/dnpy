import numpy as np


def to_categorical(dataset, num_classes):
    tmp = np.zeros((len(dataset), num_classes))
    tmp[np.arange(len(dataset)), dataset] = 1.0
    return tmp
