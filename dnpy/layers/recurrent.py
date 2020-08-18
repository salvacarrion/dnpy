import copy
import numpy as np
from dnpy.layers import Layer, Tanh, Reshape
from dnpy import initializers, utils


class RNNCell:
    def __init__(self, params, grads, name="RNNCell"):
        self.params = params
        self.grads = grads

    def forward(self, x_t, a_t):
        # Activation
        a_t_hidden = np.dot(a_t, self.params['waa'].T)
        a_t_input = np.dot(x_t, self.params['wax'].T)
        a_t = np.tanh(a_t_hidden + a_t_input + self.params['ba'])

        # Output
        y_t = np.dot(a_t, self.params['wya'].T) + self.params['by']

        return y_t, a_t

    def backward(self, x_t, y_t, a_t_prev, a_t, delta_y_t, delta_a_t):
        # Activation
        self.grads['waa'] += np.outer(delta_a_t, a_t_prev)
        self.grads['wax'] += np.outer(delta_a_t, x_t)
        self.grads['ba'] += delta_a_t * 1.0

        # Output
        self.grads['wya'] += delta_y_t * a_t
        self.grads['by'] += delta_y_t * 1.0

        # Deltas (input)
        delta_x_t = np.dot(delta_y_t * (1 - y_t ** 2), self.params['wax'])  # for tanh
        delta_a_t = np.dot(delta_a_t * (1 - a_t ** 2), self.params['wya'])  # for tanh

        return delta_x_t, delta_a_t


class BaseRNN(Layer):

    def __init__(self, l_in, units, cell, params, grads,
                 kernel_initializer=None, recurrent_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 return_sequences=False, return_state=False,
                 stateful=False, unroll=False, name="BaseRNN"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Basic properties
        self.states_h = []  # [batch_size, state_size]
        self.sequences = []  # [batch_size, timesteps, output_size]
        self.units = units
        self.params = params
        self.grads = grads
        self.cell = cell
        self.stateful = stateful
        self.unroll = unroll
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Get output
        timesteps, features = l_in.oshape
        self.oshape = (units,)

        # Initialization: param
        if kernel_initializer is None:
            self.kernel_initializer = initializers.RandomUniform()
        else:
            self.kernel_initializer = kernel_initializer

        # Initialization: recurrent
        if recurrent_initializer is None:
            self.recurrent_initializer = initializers.RandomUniform()
        else:
            self.recurrent_initializer = recurrent_initializer

        # Initialization: bias
        if bias_initializer is None:
            self.bias_initializer = initializers.Zeros()
        else:
            self.bias_initializer = kernel_initializer

        # Add regularizers
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        # Unroll net
        self.sublayers = []
        if unroll:
            raise NotImplementedError("Unrolled layer not yet implemented")
            #
            # # Ugly trick to support the stateful option with the unrolled network
            # if stateful:
            #     steps = self.parents[0].batch_size * timesteps
            #     l_x_in = Reshape(l_in, shape=(1, steps, features), include_batch=True)
            #     self.sublayers.append(l_x_in)
            # else:
            #     steps = timesteps
            #     l_x_in = l_in
            #
            # l_cell_prev = None
            # for i in range(steps):
            #     l_cell_prev = RNNCell(l_x_in=l_x_in, l_a_prev=l_cell_prev, time=i,
            #                           cell=self.cell, params=self.params, grads=self.grads)
            #     self.sublayers.append(l_cell_prev)


class SimpleRNN(BaseRNN):

    def __init__(self, l_in, units,
                 kernel_initializer=None, recurrent_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 return_sequences=False, return_state=False,
                 stateful=False, unroll=False, name="SimpleRNN"):

        # Params and grads
        timesteps, features = l_in.oshape
        self.params = {'waa': np.zeros((units, units)),
                       'wax': np.zeros((units, features)),
                       'ba': np.zeros((1, units)),
                       'wya': np.zeros((units, units)),
                       'by': np.zeros((1, units))
                       }
        self.grads = {'waa': np.zeros_like(self.params['waa']),
                      'wax': np.zeros_like(self.params['wax']),
                      'ba': np.zeros_like(self.params['ba']),
                      'wya': np.zeros_like(self.params['wya']),
                      'by': np.zeros_like(self.params['by']),
                      }

        # Save states
        self.outputs, self.states_h = [], []
        self.y_t_deltas, self.a_t_deltas = [], []

        # Must be after the "self.cell" variable
        super().__init__(l_in=l_in, units=units, cell=RNNCell, params=self.params, grads=self.grads,
                         kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer,
                         bias_regularizer=bias_regularizer,
                         return_sequences=return_sequences, return_state=return_state,
                         stateful=stateful, unroll=unroll, name=name)

    def initialize(self, optimizer=None):
        super().initialize(optimizer=optimizer)

        # Initialize params
        self.kernel_initializer.apply(self.params, ['wya'])
        self.recurrent_initializer.apply(self.params, ['waa'])
        self.recurrent_initializer.apply(self.params, ['wax'])
        self.bias_initializer.apply(self.params, ['ba'])
        self.bias_initializer.apply(self.params, ['by'])

    def forward(self):
        batch_size = len(self.parents[0].output)

        self.outputs, self.states_h = [], []
        shared_cell = self.cell(params=self.params, grads=self.grads)
        if self.stateful:  # Share activation between batches
            pass
        else:
            for b in range(batch_size):
                tmp_sequences, tmp_states_h = [], []
                a_t = np.zeros((1, self.units))
                sample_b = np.array(self.parents[0].output[b])  # From list to array
                timesteps, features = sample_b.shape

                for t in range(timesteps):
                    x_t = sample_b[t]
                    y_t, a_t = shared_cell.forward(x_t, a_t)

                    # Add outputs/states
                    tmp_states_h.append(a_t)
                    tmp_sequences.append(y_t)

                # Concatenate outputs/states across timesteps (y_t, a_t)
                self.states_h.append(np.concatenate(tmp_states_h, axis=0))
                self.outputs.append(np.concatenate(tmp_sequences, axis=0))

        # Concatenate outputs/states across batches (y_t, a_t)
        self.states_h = np.stack(self.states_h, axis=0)
        self.outputs = np.stack(self.outputs, axis=0)

        # Return sequence?
        if self.return_sequences:
            self.output = self.outputs
        else:
            self.output = self.outputs[:, -1, :]
        asdas = 33

    def backward(self):
        batch_size = len(self.parents[0].output)

        self.x_t_deltas, self.a_t_deltas = [], []
        shared_cell = self.cell(params=self.params, grads=self.grads)
        if self.stateful:  # Share activation between batches
            pass
        else:
            for b in range(batch_size):
                tmp_x_t_deltas, tmp_a_t_deltas = [], []
                x_b = self.parents[0].output[b]
                y_b = self.output[b]
                states_h_b = self.states_h[b]

                delta_y_t_b = np.expand_dims(self.delta[b], axis=0)
                delta_a_t_b = np.ones_like(self.params['ba'])  # Given param
                delta_a_t = delta_a_t_b

                timesteps, features = x_b.shape
                for t in reversed(range(timesteps)):
                    x_t = np.expand_dims(x_b[t], axis=0)
                    y_t = np.expand_dims(y_b[t], axis=0) if self.return_sequences else np.expand_dims(y_b, axis=0)
                    a_t = np.expand_dims(states_h_b[t], axis=0)
                    a_t_prev = np.expand_dims(states_h_b[t-1], axis=0) if t > 0 else np.zeros((1, self.units))

                    if self.return_sequences:
                        delta_y_t = delta_y_t_b[t]
                    else:
                        if t == timesteps-1:
                            delta_y_t = delta_y_t_b
                        else:
                            delta_y_t = np.ones_like(self.params['by'])

                    delta_x_t, delta_a_t = shared_cell.backward(x_t, y_t, a_t_prev, a_t, delta_y_t, delta_a_t)

                    # Add outputs/states
                    tmp_x_t_deltas.append(delta_x_t)
                    tmp_a_t_deltas.append(delta_a_t)

                # Concatenate outputs/states across timesteps (x_t, a_t)
                self.x_t_deltas.append(np.concatenate(tmp_x_t_deltas, axis=0))
                self.a_t_deltas.append(np.concatenate(tmp_a_t_deltas, axis=0))

        # Concatenate outputs/states across batches (x_t, a_t)
        self.x_t_deltas = np.stack(self.x_t_deltas, axis=0)
        self.a_t_deltas = np.stack(self.a_t_deltas, axis=0)

        self.parents[0].delta = self.x_t_deltas
        asdas = 33