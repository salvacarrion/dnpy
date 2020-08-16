import copy
import numpy as np
from dnpy.layers import Layer
from dnpy import initializers, utils


class Input(Layer):

    def __init__(self, shape, batch_size=None, name="Input"):
        super().__init__(name=name)
        self.oshape = shape
        self.batch_size = batch_size

    def forward(self):
        pass

    def backward(self):
        pass


class Embedding(Layer):
    """
    input_dim: vocabulary size
    output_dim: embedding latent space
    input_length: length of each sentence
    """
    def __init__(self, l_in, input_dim, output_dim, input_length, embeddings_initializer=None, embeddings_regularizer=None, name="Embedding"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.oshape = (input_length, output_dim)

        # Params and grads
        self.params = {'w1': np.zeros((input_dim, output_dim))}
        self.grads = {'w1': np.zeros_like(self.params['w1'])}

        # Initialization: param
        if embeddings_initializer is None:
            self.embeddings_initializer = initializers.RandomUniform()
        else:
            self.embeddings_initializer = embeddings_initializer

        # Add regularizers
        self.embeddings_regularizer = embeddings_regularizer

    def initialize(self, optimizer=None):
        super().initialize(optimizer=optimizer)

        # Initialize params
        self.embeddings_initializer.apply(self.params, ['w1'])

    def forward(self):
        # Get embeddings corresponding to word indices (for each sample)
        word_indices = self.parents[0].output
        sentence = self.params['w1'][word_indices]
        self.output = sentence  # (batch_size, input_length, self.output_dim)

    def backward(self):
        # Update embeddings corresponding to word indices
        word_indices = self.parents[0].output
        self.grads['w1'][word_indices] += np.mean(self.delta, axis=0)


class Dense(Layer):

    def __init__(self, l_in, units, kernel_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, bias_regularizer=None, name="Dense"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Check layer compatibility
        if len(l_in.oshape) != 1:
            raise ValueError(f"Expected a 1D layer ({self.name})")

        # Tricky
        if units == -1:
            units = l_in.oshape[0]

        self.oshape = (units,)
        self.units = units

        # Params and grads
        self.params = {'w1': np.zeros((self.parents[0].oshape[0], self.units)),
                       'b1': np.zeros((1, self.units))}
        self.grads = {'w1': np.zeros_like(self.params['w1']),
                      'b1': np.zeros_like(self.params['b1'])}

        # Initialization: param
        if kernel_initializer is None:
            fan_in, fan_out = self.params['w1'].shape
            self.kernel_initializer = initializers.HeNormal(fan_in=fan_in, fan_out=fan_out)
        else:
            self.kernel_initializer = kernel_initializer

        # Initialization: bias
        if bias_initializer is None:
            self.bias_initializer = initializers.Zeros()
        else:
            self.bias_initializer = kernel_initializer

        # Add regularizers
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def initialize(self, optimizer=None):
        super().initialize(optimizer=optimizer)

        # Initialize params
        self.kernel_initializer.apply(self.params, ['w1'])
        self.bias_initializer.apply(self.params, ['b1'])

    def forward(self):
        self.output = np.dot(self.parents[0].output, self.params['w1']) + self.params['b1']

    def backward(self):
        # Each layer sets the delta of their parent (m,13)=>(m, 10)=>(m, 1)=>(1,1)
        self.parents[0].delta = np.dot(self.delta, self.params['w1'].T)

        # Compute gradients
        m = self.output.shape[0]
        g_w1 = np.dot(self.parents[0].output.T, self.delta)
        g_b1 = np.sum(self.delta, axis=0, keepdims=True)

        # Add regularizers (if needed)
        if self.kernel_regularizer:
            g_w1 += self.kernel_regularizer.backward(self.params['w1'])
        if self.bias_regularizer:
            g_b1 += self.bias_regularizer.backward(self.params['b1'])

        self.grads['w1'] += g_w1/m
        self.grads['b1'] += g_b1/m


class Reshape(Layer):

    def __init__(self, l_in, shape, include_batch=False, name="Reshape"):
        super().__init__(name=name)
        self.parents.append(l_in)
        self.include_batch = include_batch

        if include_batch:
            self.oshape = shape[1:]
            self.oshape_batch = shape
        else:
            # Check if shape is inferred
            if shape == -1 or shape[0] == -1:
                shape = (int(np.prod(self.parents[0].oshape)),)

            # Check layer compatibility
            if np.prod(self.parents[0].oshape) != np.prod(shape):
                raise ValueError(f"Not compatible shapes ({self.name})")

            self.oshape = shape
            self.oshape_batch = None

    def forward(self):
        new_shape = (-1, *self.oshape) if not self.include_batch else self.oshape_batch
        self.output = np.reshape(self.parents[0].output, newshape=new_shape)

    def backward(self):
        new_shape = (-1, *self.parents[0].oshape) if not self.include_batch else self.oshape_batch
        self.parents[0].delta = np.reshape(self.delta, newshape=new_shape)


class Flatten(Reshape):

    def __init__(self, l_in, name="Flatten"):
        super().__init__(l_in, shape=(-1,), name=name)


