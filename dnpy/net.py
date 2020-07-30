import math
import numpy as np
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
            assert x_i.shape[0] == y_i.shape[0]  # num. samples

    # Only test
    if x_test and y_test:
        # Check number of samples (x/y test)
        for x_i, y_i in zip(x_test, y_test):
            assert x_i.shape[0] == y_i.shape[0]  # num. samples


def get_layers(net):
    # Get all layers in a single list
    vertices = []
    visited = set()

    queue = list(net.l_out)  # shallow copy
    while queue:  # Might be multiple outputs
        node = queue.pop(0)  # Pop first element
        _id = id(node)

        # Check if this layer has already been explored
        if _id not in visited:
            visited.add(_id)
            vertices.append(node)

            # Add new layers to the queu
            for node_parent in node.parents:
                queue.append(node_parent)

    return vertices


def topological_sort_helper(i, nodes, visited, stack):
    # Mark the current node as visited.
    visited[i] = True

    # Walk through each node adjacent to this vertex
    for node in nodes[i].parents:
        if not visited[node.index]:
            topological_sort_helper(node.index, nodes, visited, stack)

    # Push current vertex to stack which stores result
    stack.insert(0, nodes[i])


def topological_sort(net):
    # Topologial sort
    nodes = get_layers(net)
    visited = [False] * len(nodes)
    stack = []

    # Set index in each layer
    for i, node in enumerate(nodes):
        node.index = i

    # Recursive call for each vertex
    for node in nodes:
        if not visited[node.index]:
            topological_sort_helper(node.index, nodes, visited, stack)
    return stack


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
                str_l += ["{}={:.5f}".format(self.losses[i].name, float(losses[i][0]))]

        # Metrics
        if metrics is not None:
            assert len(self.l_out) == len(metrics)
            assert len(self.metrics) == len(metrics[0])

            for i in range(len(self.l_out)):  # N metrics for M output layers
                m = []
                for j in range(len(self.metrics)):
                    m += ["{}={:.5f}".format(self.metrics[j].name, float(metrics[i][j]))]
                str_m += ["{}: ({})".format(self.l_out[i].name, ", ".join(m))]
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
        self.bts_layers = topological_sort(self)

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
                b_losses = self.compute_losses(y_target=y_train_mb)
                if (i % print_rate) == 0:
                    b_metrics = self.compute_metrics(y_target=y_train_mb)
                    str_eval = self._format_eval(b_losses, b_metrics)
                    print(f"\t - Batch {b+1}/{num_batches} - Losses[{', '.join(str_eval[0])}]; Metrics[{'; '.join(str_eval[1])}]")

                # Backward
                self.reset_grads()
                self.do_delta(b_losses)
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
                    lo, me = self.evaluate(x_test, y_test, batch_size=min(1, len(x_test)))
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
            lo = self.compute_losses(y_target=y_test_mb)
            lo = [l[0] for l in lo]  # Ignore deltas
            losses.append(lo)

            # Metrics (n per output layer)
            me = self.compute_metrics(y_target=y_test_mb)
            metrics.append(me)

        # List to array
        losses = np.array(losses)  # 2 dims (samples, losses)
        metrics = np.array(metrics)  # 3 dims (samples, metrics, outputs)

        # Compute average
        losses = np.mean(losses, axis=0, keepdims=True)
        metrics = np.mean(metrics, axis=0)
        assert losses.ndim == 2
        assert metrics.ndim == 2

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
        for l in self.fts_layers:
            l.debug = self.debug
            l.initialize(optimizer=self.optimizer)

        # Config special derivatives (speed-up)
        if self.smart_derivatives:
            for l in self.l_out:
                for lo in self.losses:
                    if isinstance(l, Softmax) and isinstance(lo, CrossEntropy):
                        print("[INFO] Using derivative speed-up: 'CrossEntropy(Softmax(z))'")
                        lo.softmax_output = True
                        l.ce_loss = True

    def do_delta(self, losses):
        for i in range(len(self.l_out)):
            self.l_out[i].delta = losses[i][1]  # (loss, delta)

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
        for i in range(len(self.losses)):  # 1 loss per output layer
            y_pred_i = self.l_out[i].output
            y_target_i = y_target[i]

            loss = self.losses[i].compute_loss(y_pred_i, y_target_i)
            delta = self.losses[i].compute_delta(y_pred_i, y_target_i)
            losses.append((loss, delta))

            # Check for errors
            if math.isnan(loss):
                raise ValueError("NaNs in the loss function")
            elif math.isinf(loss):
                raise ValueError("Inf in the loss function")
            elif np.isnan(delta).any():
                raise ValueError("NaNs in the delta")
            elif np.isinf(delta).any():
                raise ValueError("Info in the delta")

        return losses

    def compute_metrics(self, y_target):
        metrics = []
        for i in range(len(self.l_out)):  # N metrics for M output layers
            metrics_l_out = []
            for j in range(len(self.metrics)):
                y_pred_i = self.l_out[i].output
                y_target_i = y_target[i]
                me = self.metrics[j].compute_metric(y_pred_i, y_target_i)
                metrics_l_out.append(me)
            metrics.append(metrics_l_out)
        return metrics

    def summary(self, batch_size=1):
        print('==================================')
        print("Model summary")
        print('==================================')

        # Set mode
        previous_mode = self.mode
        self.set_mode('test')

        # Feed random input to input layers
        x_train_mb = []
        for i in range(len(self.l_in)):
            x = np.random.random([batch_size, *self.l_in[i].oshape])
            x_train_mb.append(x)

        # Feed network
        self.feed_input(x_train_mb)

        # Forward
        self.forward()

        for i, l in enumerate(self.fts_layers):
            ishapes = []
            for l_in in l.parents:
                ishapes.append(l_in.output.shape)
            ishapes = ishapes if len(ishapes) > 1 else l.output.shape  # other / input
            print(f"#{i+1}:\t{l.name}\t\t-\t{ishapes}\t=>\t{l.output.shape}")

        print('')

        # Set previous mode
        self.set_mode(previous_mode)
