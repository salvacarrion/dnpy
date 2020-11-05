import copy
import math
import numpy as np
import pickle

from dnpy.layers import Softmax
from dnpy.losses import CrossEntropy


def check_datasets(x_train, y_train, x_test, y_test):
    # Both train and test
    if x_train and y_train and x_test and y_test:
        # Check number of inputs (NOT samples)
        assert len(x_train) == len(x_test)
        assert len(y_train) == len(y_test)

        # Check dimensions of inputs X (train/test)
        for tr_i, ts_i in zip(x_train, x_test):
            assert tr_i.shape[1:] == ts_i.shape[1:]  # Ignore samples

        # Check dimensions of inputs Y (train/test)
        for tr_i, ts_i in zip(y_train, y_test):
            assert tr_i.shape[1:] == ts_i.shape[1:]  # Ignore samples

    # Only train
    if x_train and y_train:
        # Check number of samples [x1, x2,...] for each input [X1, X2,...]
        for x_i, y_i in zip(x_train, y_train):
            assert len(x_i) == len(y_i)  # num. samples

    # Only test
    if x_test and y_test:
        # Check number of samples (x/y test)
        for x_i, y_i in zip(x_test, y_test):
            assert len(x_i) == len(y_i)  # num. samples


def bts_helper(node, mark_temp, mark_permanent, stack):
    _id = id(node)

    if _id in mark_permanent:
        return

    if _id in mark_temp:
        raise RecursionError("Not a DAG")

    # Mark as temporary
    mark_temp.add(_id)

    for pnode in node.parents:
        bts_helper(pnode, mark_temp, mark_permanent, stack)

    # Remove from temp, add to permanent
    mark_temp.remove(_id)
    mark_permanent.add(_id)

    # Add sorted layer
    stack.insert(0, node)


class Net:

    def __init__(self):
        self.l_in = None
        self.l_out = None
        self.fts_layers = None
        self.bts_layers = None
        self.optimizer = None
        self.losses = None
        self.metrics = None
        self.debug = False
        self.mode = 'test'
        self.smart_derivatives = None

    def build(self, l_in, l_out, optimizer, losses, metrics, debug=False, smart_derivatives=False):
        self.l_in = l_in
        self.l_out = l_out
        self.fts_layers = []
        self.bts_layers = []
        self.optimizer = optimizer
        self.losses = losses
        self.metrics = metrics
        self.debug = debug
        self.smart_derivatives = smart_derivatives

        # Topological sort
        self.bts()
        self.fts()

        # Initialize params
        self.initialize()

    def _format_eval(self, losses=None, metrics=None):
        str_l = []
        str_m = []

        # Losses
        if losses is not None:
            assert len(self.l_out) == len(self.losses) == len(losses)

            for i in range(len(self.losses)):
                str_l += ["{}={:.5f}".format(self.losses[i].name, float(losses[i]))]

        # Metrics
        if metrics is not None:
            assert len(self.l_out) == len(metrics)

            for i in range(len(self.l_out)):  # N metrics for M output layers
                tmp = []
                for j in range(len(self.metrics[i])):
                    tmp += ["{}={:.5f}".format(self.metrics[i][j].name, float(metrics[i][j]))]
                str_m += ["{}: ({})".format(self.l_out[i].name, ", ".join(tmp))]
        return [str_l, str_m]

    def get_minibatch(self, x_train, y_train, batch_i, batch_size):
        x_train_mb = []
        y_train_mb = []

        # Select mini-batches
        for j in range(len(self.l_in)):
            idx = batch_i * batch_size
            x_train_mb += [x_train[j][idx:idx + batch_size]]
            y_train_mb += [y_train[j][idx:idx + batch_size]]

        return x_train_mb, y_train_mb

    def fts(self):
        self.fts_layers = list(self.bts_layers)
        self.fts_layers.reverse()

    def bts(self):
        # Topological sort
        self.bts_layers = []
        mark_temp = set()
        mark_permanent = set()

        # Walk from outputs to inputs (there are parents but not childs)
        for node in self.l_out:  # Might be multiple outputs
            _id = id(node)

            # Check if this layer has already been explored
            if _id not in mark_permanent:
                bts_helper(node, mark_temp, mark_permanent, self.bts_layers)

        assert not mark_temp

    def fit(self, x_train, y_train, batch_size, epochs, x_test=None, y_test=None, evaluate_epoch=True, print_rate=1):
        # Check input/output compatibility
        assert isinstance(x_train, list)
        assert isinstance(y_train, list)
        assert isinstance(x_test, list) or x_test is None
        assert isinstance(y_test, list) or y_test is None
        assert len(self.l_in) == len(x_train)
        assert len(self.l_out) == len(self.losses) == len(y_train)

        # Check datasets compatibility
        check_datasets(x_train, y_train, x_test, y_test)

        # Get basic data
        batch_size = int(batch_size)
        num_samples = len(x_train[0])
        num_batches = int(num_samples/batch_size)

        # Train model
        for i in range(epochs):
            if (i % print_rate) == 0:
                print(f"Epoch {i+1}/{epochs}...")

            # Set mode
            if self.mode != "train":
                self.set_mode('train')

            # Train mini-batches
            for b in range(num_batches):
                # Get minibatch
                x_train_mb, y_train_mb = self.get_minibatch(x_train, y_train, b, batch_size)

                # Feed network
                self.feed_input(x_train_mb)

                # Do forward pass
                self.forward()

                # Compute loss and metrics
                b_losses, b_deltas = self.compute_losses(y_target=y_train_mb)
                if (i % print_rate) == 0:
                    b_metrics = self.compute_metrics(y_target=y_train_mb)
                    str_eval = self._format_eval(b_losses, b_metrics)
                    print(f"\t - Batch {b+1}/{num_batches} - Losses[{', '.join(str_eval[0])}]; Metrics[{'; '.join(str_eval[1])}]")

                # Backward
                self.reset_grads()
                self.do_delta(b_deltas)
                self.backward()
                self.apply_grads()

            if evaluate_epoch and (i % print_rate) == 0:
                # Set mode
                self.set_mode('test')

                # Evaluate train
                lo, me = self.evaluate(x_train, y_train, batch_size=1)
                str_eval = self._format_eval(lo, me)
                print(f"\t - Training losses[{', '.join(str_eval[0])}]; Training metrics[{'; '.join(str_eval[1])}];")

                # Evaluate test
                if x_test is not None and y_test is not None:
                    lo, me = self.evaluate(x_test, y_test, batch_size=1)
                    str_eval = self._format_eval(lo, me)
                    print(f"\t - Validation losses[{', '.join(str_eval[0])}]; Validation metrics[{'; '.join(str_eval[1])}];")

    def evaluate(self, x_test, y_test, batch_size=1):
        # Check input/output compatibility
        assert isinstance(x_test, list)
        assert isinstance(y_test, list)
        assert len(self.l_in) == len(x_test)
        assert len(self.l_out) == len(self.losses) == len(y_test)

        # Check datasets compatibility
        check_datasets(None, None, x_test, y_test)

        # Get basic data
        num_samples = len(x_test[0])
        num_batches = int(num_samples / batch_size)
        assert batch_size <= num_samples

        # Set mode
        if self.mode != "test":
            print("[WARNING] Setting net mode to 'test'")
            self.set_mode('test')

        # Set vars
        losses = []
        metrics = []

        # Evaluate model
        for b in range(num_batches):
            # Get mini-batch
            x_test_mb, y_test_mb = self.get_minibatch(x_test, y_test, b, batch_size)

            # Feed network
            self.feed_input(x_test_mb)

            # Do forward pass
            self.forward()

            # Losses (one per output layer)
            lo, de = self.compute_losses(y_target=y_test_mb)
            losses.append(lo)

            # Metrics (n per output layer)
            me = self.compute_metrics(y_target=y_test_mb)
            metrics.append(me)

        # List to array
        losses = np.array(losses)  # 2 dims (samples, losses)
        metrics = [np.array(row) for row in list(zip(*metrics))]  # Transpose and convert to a list of arrays

        # Compute average
        losses = np.mean(losses, axis=0)
        metrics = [np.mean(me, axis=0) for me in metrics]
        assert len(losses) == len(self.losses)
        assert len(metrics) == len(self.losses)

        return losses, metrics

    def set_mode(self, mode="train"):
        self.mode = mode
        for l in self.fts_layers:
            if self.mode == "train":
                l.training = True
            elif self.mode == "test":
                l.training = False
            else:
                raise KeyError("Unknown mode")

    def initialize(self):
        batch_size = self.fts_layers[0].batch_size
        for l in self.fts_layers:
            l.debug = self.debug
            l.initialize(optimizer=self.optimizer)

            # Propagate variables
            l.batch_size = batch_size  # Might be needed (eg. special case RNN)

        # Config special derivatives (speed-up)
        if self.smart_derivatives:
            for l in self.l_out:
                for lo in self.losses:
                    if isinstance(l, Softmax) and isinstance(lo, CrossEntropy):
                        print("[INFO] Using derivative speed-up: 'CrossEntropy(Softmax(z))'")
                        lo.softmax_output = True
                        l.ce_loss = True

    def do_delta(self, deltas):
        for i in range(len(self.l_out)):
            self.l_out[i].delta = deltas[i]

    def reset_grads(self):
        for l in self.fts_layers:
            for k in l.grads.keys():
                l.grads[k].fill(0.0)

    def feed_input(self, x):
        # Feed batch into the network
        for j in range(len(self.l_in)):
            self.l_in[j].output = x[j]

    def forward(self):
        for l in self.fts_layers:
            l.forward()
            if self.debug:
                l.print_stats()

    def backward(self):
        for l in self.bts_layers:
            l.backward()
            if self.debug:
                l.print_stats()

    def apply_grads(self):
        for l in self.bts_layers:
            # Check if the layer is frozen
            if not l.frozen:
                l.optimizer.apply(l.params, l.grads)

    def compute_losses(self, y_target):
        losses = []
        deltas = []
        for i in range(len(self.losses)):  # 1 loss per output layer
            y_pred_i = self.l_out[i].output
            y_target_i = y_target[i]

            # Save losses
            loss = self.losses[i].compute_loss(y_pred_i, y_target_i)
            losses.append(loss)

            # Save deltas
            delta = self.losses[i].compute_delta(y_pred_i, y_target_i)
            deltas.append(delta)

            # Check for errors
            if math.isnan(loss):
                raise ValueError("NaNs in the loss function")
            elif math.isinf(loss):
                raise ValueError("Inf in the loss function")
            elif np.isnan(delta).any():
                raise ValueError("NaNs in the delta")
            elif np.isinf(delta).any():
                raise ValueError("Info in the delta")

        return losses, deltas

    def compute_metrics(self, y_target):
        metrics = []
        for i in range(len(self.l_out)):  # N metrics for M output layers
            metrics_l_out = []
            for me in self.metrics[i]:
                y_pred_i = self.l_out[i].output
                y_target_i = y_target[i]
                values = me.compute_metric(y_pred_i, y_target_i)
                metrics_l_out.append(values)
            metrics.append(metrics_l_out)
        return metrics

    def summary(self):
        total_params = 0
        total_tr_params = 0
        total_notr_params = 0
        line_length = 75

        print('='*line_length)
        print("Model summary")
        print('='*line_length)
        print("{:<20} {:<20} {:<20} {:<10}".format("Layer (type)", "Input Shape/s", "Output Shape", "Param #"))
        print('-'*line_length)
        for i, l in enumerate(self.fts_layers):

            # Get shapes
            ishapes = []
            for l_in in l.parents:
                ishapes.append(l_in.oshape)
            ishapes = ishapes if len(ishapes) > 1 else l.oshape  # other / input
            oshape = l.oshape

            # Get params
            tr_params, notr_params = l.get_num_params()
            params = tr_params + notr_params
            total_tr_params += tr_params
            total_notr_params += notr_params
            total_params += params

            # Print line
            print("{:<20} {:<20} {:<20} {:<10}".format(str(l.name), str(ishapes), str(oshape), str(params)))

        print('-'*line_length)
        print('Total params: {:,}'.format(total_params))
        print('Trainable params: {:,}'.format(total_tr_params))
        print('Non-trainable params: {:,}'.format(total_notr_params))
        print('-'*line_length)
        print('')

    def get_params(self, only_trainable=False):
        params = []
        for i, l in enumerate(self.fts_layers, 0):
            layer_params = l.params
            if only_trainable:  # Copy only if it's trainable (used in grad check, batchnorm,...)
                layer_params = {k: v for k, v in l.params.items() if k in l.grads}
            p = copy.deepcopy(layer_params)
            params.append(p)
        return params

    def set_params(self, params, only_trainable=False):
        assert len(self.fts_layers) == len(params)
        for i, l in enumerate(self.fts_layers, 0):
            if only_trainable:
                for k, v in l.params.items():
                    if k in params[i]:  # Add new only if exists (used in grad check, batchnorm,...)
                        l.params[k] = params[i][k]
            else:
                assert len(l.params) == len(params[i])
                l.params = params[i]

    def get_grads(self):
        grads = []
        for i, l in enumerate(self.fts_layers, 0):
            g = copy.deepcopy(l.grads)
            grads.append(g)
        return grads

    def set_grads(self, grads):
        assert len(self.fts_layers) == len(grads)
        for i, l in enumerate(self.fts_layers, 0):
            assert len(l.grads) == len(grads[i])
            l.grads = grads[i]

    def load(self, filename):
        # Load data
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)

        # Set data
        self.set_params(data['params'])
        grads = data.get('grads')
        if grads:
            self.set_grads(grads)
        print("Model loaded!")

    def save(self, filename, save_grads=False):
        data = {
            'params': self.get_params(),
        }

        if save_grads:
            data['grads'] = self.get_grads()

        # Save data
        with open(filename, 'wb') as fp:
            pickle.dump(data, fp)
        print("Model saved!")
