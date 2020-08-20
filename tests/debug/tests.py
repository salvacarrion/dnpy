# -*- coding: utf-8 -*-

import numpy as np

from dnpy.layers.recurrent import RNNCell

hidden_dim = 3
output_dim = 4
features = 4
params = {'waa': np.zeros((hidden_dim, hidden_dim)),
               'wax': np.zeros((hidden_dim, features)),
               'ba': np.zeros((1, hidden_dim)),
               'wya': np.zeros((output_dim, hidden_dim)),
               'by': np.zeros((1, output_dim))
               }
grads = {'waa': np.zeros_like(params['waa']),
              'wax': np.zeros_like(params['wax']),
              'ba': np.zeros_like(params['ba']),
              'wya': np.zeros_like(params['wya']),
              'by': np.zeros_like(params['by']),
              }


params['wax'] = np.array([[0.6, 0.8, 0.4, 0.8],
                            [0.2, 0.2, 0.8, 0.7],
                            [0.9, 0.8, 0.1, 0.2]])
params['waa'] = np.array([[0.1, 0.5, 0.1],
                          [0.5, 0.9, 0.3],
                          [0.3, 0.2, 0.1]])

mycell = RNNCell(params, grads)

x_t = np.array([[1, 0, 0, 0]])

a_t_prev = np.array([[0, 0, 0]])
y_t, a_t = mycell.forward(x_t, a_t_prev)
asdas = 33