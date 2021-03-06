import copy
import numpy as np
import random

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


def show_img(img):
    plt.imshow(img)
    plt.show()


def get_padding(padding, input_size, output_size, kernel_size, strides):
    # Note: Padding along height/width

    # Convert to numpy
    input_size = np.array(input_size)
    output_size = np.array(output_size)
    kernel_size = np.array(kernel_size)
    strides = np.array(strides)

    # Select mode
    padding = str(padding).lower()
    if padding in {"none", "valid"}:
        pads = np.zeros_like(kernel_size)
    elif padding in {"same", "zeros"}:
        pads = np.ceil((strides*(output_size-1)-input_size+kernel_size) )  # along axis
        # pads2 = np.zeros_like(kernel_size)
        # for i in range(len(pads2)):
        #     if input_size[i] % strides[i] == 0:
        #         pads2[i] = max(kernel_size[i] - strides[i], 0)
        #     else:
        #         pads2[i] = max(kernel_size[i] - (input_size[i] % strides[i]), 0)
        # assert np.all(pads == pads2)
    else:
        raise ValueError("Unknown padding")
    return pads.astype(int)


def get_side_paddings(pads):
    side_pads = []

    # Specific paddings (leave more at the bottom/right)
    for p in pads:
        pad1 = p // 2
        pad2 = p - pad1
        side_pads.append((pad1, pad2))

    return tuple(side_pads)


def get_output(input_size, kernel_size, strides, padding, dilation_rate=None):
    # Convert to numpy
    input_size = np.array(input_size)
    kernel_size = np.array(kernel_size)
    strides = np.array(strides)
    dilation_rate = np.array(dilation_rate) if dilation_rate else np.ones_like(kernel_size)
    assert input_size.ndim == kernel_size.ndim == strides.ndim == dilation_rate.ndim

    if padding in {"none", "valid"}:
        output = np.ceil((input_size - dilation_rate*(kernel_size - 1)) / strides)
    elif padding in {"same", "zeros"}:
        output = np.ceil(input_size/strides)
    elif isinstance(padding, list) or isinstance(padding, tuple) or isinstance(padding, np.ndarray):
        padding = np.array(padding)  # single padding
        output = np.floor((input_size - kernel_size + 2 * padding) / strides) + 1  # Typical formula
    else:
        raise ValueError("Unknown padding")

    return output.astype(int)


def params2vector(params):
    vector = []
    pi = 0
    params_slides = []
    for li in range(len(params)):
        for kp, vp in params[li].items():
            vector.append(vp.reshape(-1, 1))
            params_slides.append((pi, pi+vp.size))
            pi += vp.size

    vector = np.concatenate(vector, axis=0)
    return vector, params_slides


def vector2params(vector, params):
    pi = 0
    new_params = copy.deepcopy(params)
    for li in range(len(params)):
        for kp, vp in params[li].items():
            new_vp = vector[pi:pi+vp.size]
            new_vp = np.reshape(new_vp, vp.shape)
            new_params[li][kp] = new_vp
            pi += vp.size
    return new_params


def sample_params_slices(params_slides, max_samples, flat=True):
    indices = []
    for start_idx, end_idx in params_slides:
        size = end_idx - start_idx
        k = min(size, max_samples)
        rdn_indices = random.sample(range(start_idx, end_idx), k)
        if flat:
            indices.extend(rdn_indices)
        else:
            indices.append(rdn_indices)
    return indices
