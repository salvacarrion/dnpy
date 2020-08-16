import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy.regularizers import *
from dnpy import metrics, losses
from dnpy import utils

# For debugging
np.random.seed(42)


def noisy_sin(steps_per_cycle=50,
              number_of_cycles=500,
              random_factor=0.4):
    '''
    random_factor    : amont of noise in sign wave. 0 = no noise
    number_of_cycles : The number of steps required for one cycle

    Return :
    pd.DataFrame() with column sin_t containing the generated sin wave
    '''
    np.random.seed(0)
    df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
    df["sin_t"] = df.t.apply(
        lambda x: math.sin(x * (2 * math.pi / steps_per_cycle) + np.random.uniform(-1.0, +1.0) * random_factor))
    df["sin_t_clean"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)))
    print("create period-{} sin wave with {} cycles".format(steps_per_cycle, number_of_cycles))
    print("In total, the sin wave time series length is {}".format(steps_per_cycle * number_of_cycles + 1))
    return (df)


def _load_data(data, n_prev=100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].to_numpy())  # X = [1,2,3, ?]
        docY.append(data.iloc[i+n_prev].to_numpy())  # Y = 4
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


def main():
    steps_per_cycle = 10
    df = noisy_sin(steps_per_cycle=steps_per_cycle,
                   random_factor=0)

    n_plot = 8
    df[["sin_t"]].head(steps_per_cycle * n_plot).plot(
        title="Generated first {} cycles".format(n_plot),
        figsize=(15, 3))

    c = 0.8
    tr_size = int(len(df) * c)
    length_of_sequences = 2
    df_train = df[["sin_t"]].iloc[:tr_size]
    df_test = df[["sin_t"]].iloc[tr_size:]
    (x_train, y_train) = _load_data(df_train, n_prev=length_of_sequences)
    (x_test, y_test) = _load_data(df_test, n_prev=length_of_sequences)

    # Params *********************************
    batch_size = int(len(x_train) / 1)
    epochs = 10

    # Define architecture
    l_in = Input(shape=x_train[0].shape, batch_size=batch_size)
    l = SimpleRNN(l_in, units=1, stateful=True, return_sequences=False, unroll=True)
    l_out = Dense(l, units=1)

    # Build network
    mymodel = Net()
    mymodel.build(
        l_in=[l_in],
        l_out=[l_out],
        optimizer=Adam(lr=0.01),
        losses=[losses.MSE()],
        metrics=[[metrics.MSE()]],
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

    sdfsd= 33


if __name__ == "__main__":
    main()
