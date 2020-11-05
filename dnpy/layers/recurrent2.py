import copy
import numpy as np
from dnpy.layers import Layer, Tanh, Reshape
from dnpy import initializers, utils


class RNNCell:
    def __init__(self, params, grads, name="RNNCell"):
        self.params = params
        self.grads = grads

    def forward(self, x_t, h_t_prev):
        # Activation
        a_t_hidden = np.dot(h_t_prev, self.params['waa'].T)
        a_t_input = np.dot(x_t, self.params['wax'].T)
        a_t = a_t_hidden + a_t_input + self.params['ba']
        h_t = np.tanh(a_t)

        # Output
        y_t_pre_act = np.dot(h_t, self.params['wya'].T) + self.params['by']
        y_t = np.tanh(y_t_pre_act)

        return y_t, h_t

    def backward(self, x_b, y_b, h_b, delta_y_b, delta_h_b, batch_size, t, bptt_truncate):
        pass

        # return delta_x_t, delta_a_t_prev


class BaseRNN(Layer):

    def __init__(self, l_in, hidden_dim, output_dim, cell, params, grads,
                 kernel_initializer=None, recurrent_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 return_sequences=False, return_state=False,
                 stateful=False, unroll=False, bptt_truncate=None, name="BaseRNN"):
        super().__init__(name=name)
        self.parents.append(l_in)

        # Basic properties
        self.states_h = []  # [batch_size, state_size]
        self.sequences = []  # [batch_size, timesteps, output_size]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.params = params
        self.grads = grads
        self.cell = cell
        self.stateful = stateful
        self.unroll = unroll
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.bptt_truncate = bptt_truncate

        # Get output
        self.oshape = (self.output_dim,)

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




class RNNLayer:
    def forward(self, x, prev_s, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)

class MultiplyGate:
    def forward(self,W, x):
        return np.dot(W, x.T).T
    def backward(self, W, x, dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(dz, W)
        return dW, dx

class AddGate:
    def forward(self, x1, x2):
        return x1 + x2
    def backward(self, x1, x2, dz):
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2

class Tanha:
    def forward(self, x):
        return np.tanh(x)
    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - np.square(output)) * top_diff

mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanha()

class SimpleRNN(BaseRNN):

    def __init__(self, l_in, hidden_dim, output_dim=None,
                 kernel_initializer=None, recurrent_initializer=None, bias_initializer=None,
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 return_sequences=False, return_state=False,
                 stateful=False, unroll=False, bptt_truncate=None, name="SimpleRNN"):

        # Default ouput dims
        output_dim = output_dim if output_dim else hidden_dim

        # Params and grads
        timesteps, features = l_in.oshape
        self.params = {'waa': np.zeros((hidden_dim, hidden_dim)),
                       'wax': np.zeros((hidden_dim, features)),
                       'ba': np.zeros((1, hidden_dim)),
                       'wya': np.zeros((output_dim, hidden_dim)),
                       'by': np.zeros((1, output_dim))
                       }
        self.grads = {'waa': np.zeros_like(self.params['waa']),
                      'wax': np.zeros_like(self.params['wax']),
                      'ba': np.zeros_like(self.params['ba']),
                      'wya': np.zeros_like(self.params['wya']),
                      'by': np.zeros_like(self.params['by']),
                      }

        # Save states
        self.outputs, self.states_h = [], []
        self.y_t_deltas, self.h_t_deltas = [], []
        self.delta_h_t = None

        # Must be after the "self.cell" variable
        super().__init__(l_in=l_in, hidden_dim=hidden_dim, output_dim=output_dim, cell=RNNCell,
                         params=self.params, grads=self.grads,
                         kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer,
                         bias_regularizer=bias_regularizer,
                         return_sequences=return_sequences, return_state=return_state,
                         stateful=stateful, unroll=unroll, bptt_truncate=bptt_truncate, name=name)

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
                a_t = np.zeros_like(self.params['ba'])
                sample_b = np.array(self.parents[0].output[b])  # From list to array
                timesteps, features = sample_b.shape

                prev_s = np.zeros_like(self.params['ba'])
                layers = []

                for t in range(timesteps):
                    x_t = sample_b[t]

                    layer = RNNLayer()
                    layer.forward(x_t, prev_s, self.params['wax'], self.params['waa'], self.params['wya'])
                    prev_s = layer.s
                    layers.append(layer)

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
        self.x_t_deltas, self.h_t_deltas = [], []
        shared_cell = self.cell(params=self.params, grads=self.grads)
        if self.stateful:  # Share activation between batches
            pass
        else:
            for b in range(batch_size):
                tmp_x_t_deltas, tmp_h_t_deltas = [], []
                x_b = self.parents[0].output[b]
                y_b = self.output[b]
                h_b = self.states_h[b]

                delta_y_t_b = np.expand_dims(self.delta[b], axis=0)
                if self.delta_h_t:
                    delta_h_t_prev = self.delta_h_t[b]
                else:
                    delta_h_t_prev = np.zeros_like(self.params['ba'])

                timesteps, features = x_b.shape
                bptt_truncate = self.bptt_truncate if self.bptt_truncate else timesteps
                for t in reversed(range(timesteps)):
                    x_t = np.expand_dims(x_b[t], axis=0)
                    y_t = np.expand_dims(y_b[t], axis=0) if self.return_sequences else np.expand_dims(y_b, axis=0)
                    h_t = np.expand_dims(h_b[t], axis=0)
                    h_t_prev = np.expand_dims(h_b[t-1], axis=0) if t > 0 else np.zeros_like(self.params['ba'])
                    delta_y_t = delta_y_t_b if t == timesteps - 1 else np.zeros_like(self.params['by'])

                    # Computer transfer derivative for the activations
                    dl_dy = delta_y_t  # delta d_L/d_y * d_y/d_g2
                    dy_dypre = (1 - y_t ** 2)  # y_t = tanh()
                    dypre_dh = self.params['wya']
                    dh_da = (1 - h_t ** 2)  # a_t = tanh()

                    # Activation
                    delta_y_t_pre = dl_dy * dy_dypre
                    delta_y_h_t = np.dot(delta_y_t_pre, dypre_dh)

                    delta_h_t_pre = (delta_y_h_t + delta_h_t_prev) * dh_da  # sum errors

                    # Output
                    dwya = delta_y_t_pre * h_t
                    dby = delta_y_t_pre * 1.0
                    self.grads['wya'] += dwya
                    self.grads['by'] += dby

                    # Deltas (input)
                    delta_h_t_prev = np.dot(delta_h_t_pre, self.params['waa'])
                    delta_x_t = np.dot(delta_h_t_pre, self.params['wax'])


                    cell = RNNLayer()
                    delta_h_t = np.zeros_like(self.params['ba'])
                    dprev_s, dU_t, dW_t, dV_t = cell.backward(x_t,
                                                              h_t_prev,
                                                              self.params['wax'],
                                                              self.params['waa'],
                                                              self.params['wya'],
                                                              delta_h_t,
                                                              delta_y_t)

                    dmulv = np.zeros_like(delta_y_t)
                    for i in range(t - 1, max(-1, t - bptt_truncate - 1), -1):
                        x_t_ = np.expand_dims(x_b[i], axis=0)
                        h_t_ = np.expand_dims(h_b[i - 1], axis=0) if i > 0 else np.zeros_like(self.params['ba'])

                        dprev_s, dU_i, dW_i, dV_i = cell.backward(x_t_,
                                                                  h_t_,
                                                                  self.params['wax'],
                                                                  self.params['waa'],
                                                                  self.params['wya'],
                                                                  dprev_s,
                                                                  dmulv)
                        dU_t += dU_i
                        dW_t += dW_i
                    self.grads['wya'] += dV_t
                    self.grads['wax'] += dU_t
                    self.grads['waa'] += dW_t
                    # Backpropagation through time (for at most self.bptt_truncate steps)
                    # dprev_s = delta_h_t_pre
                    # for bptt_step in reversed(range(max(0, t - bptt_truncate), t + 1)):
                    #     x_t_ = np.expand_dims(x_b[bptt_step], axis=0)
                    #     h_t_ = np.expand_dims(h_b[bptt_step - 1], axis=0) if bptt_step > 0 else np.zeros_like(self.params['ba'])
                    #
                    #     dwaa = np.dot(dprev_s.T, h_t_)
                    #     dwax = np.dot(dprev_s.T, x_t_)
                    #     dba = dprev_s * 1.0
                    #
                    #     self.grads['waa'] += dwaa
                    #     self.grads['wax'] += dwax
                    #     self.grads['ba'] += dba

                        # dprev_s = np.dot(dprev_s * (1 - h_t_ ** 2), self.params['waa'])



                    # Add outputs/states
                    tmp_h_t_deltas.append(delta_h_t_prev)
                    tmp_x_t_deltas.append(delta_x_t)

                # Set original order
                tmp_h_t_deltas.reverse()
                tmp_x_t_deltas.reverse()

                # Concatenate outputs/states across timesteps (x_t, h_t)
                self.h_t_deltas.append(np.concatenate(tmp_h_t_deltas, axis=0))
                self.x_t_deltas.append(np.concatenate(tmp_x_t_deltas, axis=0))

        # Concatenate outputs/states across batches (x_t, a_t)
        self.h_t_deltas = np.stack(self.h_t_deltas, axis=0)
        self.x_t_deltas = np.stack(self.x_t_deltas, axis=0)

        self.parents[0].delta = self.x_t_deltas
        asdas = 33