import copy
import numpy as np
from dnpy.layers import Layer, Tanh
from dnpy import initializers, utils


class RNNCell:
    def __init__(self, params, grads):
        self.params = params
        self.grads = grads

    def forward(self, x_t, a_prev):
        a_t_hidden = np.dot(a_prev, self.params['waa'].T)
        a_t_input = np.dot(x_t, self.params['wax'].T)
        a_t = np.tanh(a_t_hidden+a_t_input+self.params['ba'])
        y_t = np.dot(a_t, self.params['wya'].T) + self.params['by']
        return y_t, a_t

    def backward(self, x_t, a_prev):
        pass


class RNNUnit(Layer):

    def __init__(self, l_in, l_prev, time, cell, act, params, grads, stateful, name="RNNUnit"):
        super().__init__(name=name)
        self.parents.append(l_in)

        self.l_in = l_in
        self.l_prev = l_prev
        self.time = time
        self.cell = cell
        self.act = act
        self.params = params
        self.grads = grads
        self.stateful = stateful

        # layers
        self.l1_cell = Layer(name="dummy")
        self.l2_act = self.act(self.l1_cell)

    def forward(self):
        if self.stateful:
            raise NotImplementedError("Stateful mode is not compatible with unrolled layer")

        else:
            # Recurrent unit cell
            x_t = self.l_in.output[:, self.time]
            a_prev = self.l_prev.output if self.l_prev else 0
            y_t, a_t = self.cell.forward(x_t, a_prev)
            self.l1_cell.output = a_t

            # Activation
            self.l2_act.forward()
            l2 = self.l2_act.output

            # Output
            self.output = l2

    def backward(self):
        pass


class BaseRNN(Layer):

    def __init__(self, l_in, cell, params, grads,
                 kernel_initializer=None, recurrent_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 return_sequences=False, return_state=False,
                 stateful=False, unroll=False, name="BaseRNN"):
        super().__init__(name=name)

        # Basic properties
        self.params = params
        self.grads = grads
        self.cell = cell(self.params, self.grads)  # RNNCell,...
        self.stateful = stateful
        self.unroll = unroll
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Get output
        timesteps, features = l_in.oshape
        if self.return_sequences:
            self.oshape = (timesteps, features)
        else:
            self.oshape = (features,)

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
            l_prev = None
            for i in range(timesteps):
                l_prev = RNNUnit(l_in=l_in, l_prev=l_prev, time=i,
                                 cell=self.cell, act=Tanh, stateful=self.stateful,
                                 params=self.params, grads=self.grads)
                self.sublayers.append(l_prev)
        else:
            raise NotImplementedError("Rolled layer not yet implemented")

    def forward(self):
        # Custom forward
        tmp_outputs = []
        for l in self.sublayers:
            l.forward()

            if self.return_sequences:
                tmp_outputs.append(l.output)
            else:
                tmp_outputs = l.output

        # Check if we have to return a sequence
        if self.return_sequences:
            self.output = np.concatenate(tmp_outputs, axis=0)
        else:
            self.output = tmp_outputs
        asdas = 33

    def backward(self):
        pass


class SimpleRNN(BaseRNN):

    def __init__(self, l_in, units,
                 kernel_initializer=None, recurrent_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 return_sequences=False, return_state=False,
                 stateful=False, unroll=False, name="SimpleRNN"):

        # Params and grads
        self.units = units
        timesteps, features = l_in.oshape
        self.params = {'waa': np.zeros((self.units, self.units)),
                       'wax': np.zeros((self.units, features)),
                       'ba': np.zeros((1, self.units)),
                       'wya': np.zeros((features, self.units)),
                       'by': np.zeros((1, features))
                       }
        self.grads = {'waa': np.zeros_like(self.params['waa']),
                      'wax': np.zeros_like(self.params['wax']),
                      'ba': np.zeros_like(self.params['ba']),
                      'wya': np.zeros_like(self.params['wya']),
                      'by': np.zeros_like(self.params['by']),
                      }

        # Must be after the "self.cell" variable
        super().__init__(l_in=l_in, cell=RNNCell, params=self.params, grads=self.grads,
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

    # def forward(self):
    #     batch_size, timesteps, features = self.parents[0].output.shape
    #
    #     output = []
    #     for b in range(batch_size):
    #
    #         xstars, hs_hand = [], []
    #         a_t = np.zeros((1, self.units))
    #         for t in range(timesteps):  # This could be unrolled
    #             x_t = np.expand_dims(self.parents[0].output[b][t], axis=0)
    #             y_t, a_t = self.cell.forward(x_t, a_t)
    #             xstars.append(y_t)
    #             hs_hand.append(a_t)
    #
    #         # Check if we have to return a sequence
    #         if self.return_sequences:
    #             output.append(np.array(xstars))
    #         else:
    #             output.append(xstars[-1])  # Keep only last
    #
    #     self.output = np.concatenate(output, axis=0)
    #
    # def backward(self):
    #     batch_size, timesteps, features = self.parents[0].output.shape
    #
    #     output = []
    #     for b in range(batch_size):
    #
    #         xstars, hs_hand = [], []
    #         a_t = np.zeros((1, self.units))
    #         for t in range(timesteps):  # This could be unrolled
    #             x_t = np.expand_dims(self.parents[0].output[b][t], axis=0)
    #             y_t, a_t = self.cell.forward(x_t, a_t)
    #             xstars.append(y_t)
    #             hs_hand.append(a_t)
    #
    #         # Check if we have to return a sequence
    #         if self.return_sequences:
    #             output.append(np.array(xstars))
    #         else:
    #             output.append(xstars[-1])  # Keep only last
    #
    #     self.output = np.concatenate(output, axis=0)
