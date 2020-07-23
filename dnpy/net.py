import math
import numpy as np

class Net:

    def __init__(self):
        self.l_in = None
        self.l_out = None
        self.fts_layers = None
        self.bts_layers = None
        self.opt = None
        self.losses = None
        self.metrics = None
        self.debug = False

    def build(self, l_in, l_out, opt, losses, metrics, debug=False):
        self.l_in = l_in
        self.l_out = l_out
        self.fts_layers = []
        self.bts_layers = []
        self.opt = opt
        self.losses = losses
        self.metrics = metrics
        self.debug = debug

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
            assert len(losses) == len(self.losses)

            for i, l in enumerate(self.losses):
                str_l += ["{}={:.5f}".format(l.name, float(losses[i][0]))]

        # Metrics
        if metrics is not None:
            assert len(metrics) == len(self.metrics)

            for i, m in enumerate(self.metrics):
                str_m += ["{}={:.5f}".format(m.name, float(metrics[i]))]
        return [str_l, str_m]

    def fts(self):
        self.fts_layers = list(self.bts_layers)
        self.fts_layers.reverse()

    def bts(self):
        layer = self.l_out[0]  # TODO: Temp! Topological sort needed
        while layer.parent:
            self.bts_layers.append(layer)
            layer = layer.parent
        self.bts_layers.append(layer)  # Add last

    def fit(self, x_train, y_train, batch_size, epochs, x_test=None, y_test=None, evaluate_epoch=True,
            print_rate=1):
        assert x_train.shape[0] == y_train.shape[0]
        # assert y_train.shape[1:] == self.l_out[0].oshape
        assert batch_size <= len(x_train)

        num_samples = len(x_train)
        num_batches = int(num_samples/batch_size)

        print(f"\n========================")
        print(f"Training")
        print(f"========================")

        for i in range(epochs):
            if (i % print_rate) == 0:
                print(f"Epoch {i+1}/{epochs}...")

            for b in range(num_batches):
                # Select mini-bach
                idx = b*batch_size
                x_train_mb = x_train[idx:idx+batch_size].T
                y_train_mb = y_train[idx:idx+batch_size].T

                # Forward
                self.l_in[0].input = x_train_mb  # TODO: Temp!
                self.forward()

                # Compute loss and metrics
                b_losses = self.compute_losses(y_pred=self.l_out[0].output, y_target=y_train_mb)  # TODO: Temp!
                if (i % print_rate) == 0:
                    b_metrics = self.compute_metrics(y_pred=self.l_out[0].output, y_target=y_train_mb)  # TODO: Temp!
                    str_eval = self._format_eval(b_losses, b_metrics)
                    print(f"\t - Batch {b+1}/{num_batches} - Losses[{'; '.join(str_eval[0])}]; Metrics[{'; '.join(str_eval[1])}]")

                # Backward
                self.reset_grads()
                self.do_delta(b_losses)
                self.backward()
                self.apply_grads()

            if evaluate_epoch and (i % print_rate) == 0:
                # Evaluate train
                lo, me = self.evaluate(x_train, y_train, batch_size=1)
                str_eval = self._format_eval(lo, np.mean(me, axis=0))
                print(f"\t - Training losses[{'; '.join(str_eval[0])}]; Training metrics[{'; '.join(str_eval[1])}];")

                # Evaluate test
                if x_test is not None and y_test is not None:
                    assert x_test.shape[0] == y_test.shape[0]
                    lo, me = self.evaluate(x_test, y_test, batch_size=min(1, len(x_test)))
                    str_eval = self._format_eval(lo, me)
                    print(f"\t - Validation losses[{'; '.join(str_eval[0])}]; Validation metrics[{'; '.join(str_eval[1])}];")

    def evaluate(self, x_test, y_test, batch_size=1):
        assert x_test.shape[0] == y_test.shape[0]
        # assert y_test[0].shape == self.l_out[0].oshape
        assert batch_size <= len(x_test)

        num_samples = len(x_test)
        num_batches = int(num_samples / batch_size)

        losses = []
        metrics = []
        for b in range(num_batches):
            # Select mini-bach
            idx = b * batch_size
            x_test_mb = x_test[idx:idx + batch_size].T
            y_test_mb = y_test[idx:idx + batch_size].T

            # Forward
            self.l_in[0].input = x_test_mb  # TODO: Temp!
            self.forward()

            # Losses
            lo = self.compute_losses(y_pred=self.l_out[0].output, y_target=y_test_mb)  # TODO: Temp!
            losses.append(lo)

            # Metrics
            # Returns one average per metric, so batch must be equal to 1 to be technically correct
            me = self.compute_metrics(y_pred=self.l_out[0].output, y_target=y_test_mb)  # TODO: Temp!
            metrics.append(me)

        # [Losses] Transpose list of lists, and stack them horizontally
        tmp = []
        for lo in list(map(list, zip(*losses))):
            l = [l[0] for l in lo]  # ignore deltas
            tmp.append(np.hstack(l))
        losses = np.vstack(tmp)

        # [Metrics] Transpose list of lists, and stack them horizontally
        tmp = []
        for me in list(map(list, zip(*metrics))):
            tmp.append(np.hstack(me))
        metrics = np.vstack(tmp)

        # Compute average
        losses = np.mean(losses, axis=1, keepdims=True)
        metrics = np.mean(metrics, axis=1, keepdims=True)
        return losses, metrics

    def initialize(self):
        for l in self.fts_layers:
            l.debug = self.debug
            l.initialize()

    def do_delta(self, losses):
        for l, (loss, d_loss) in zip(self.l_out, losses):
            l.delta = d_loss

    def reset_grads(self):
        for l in self.fts_layers:
            for k in l.grads.keys():
                l.grads[k].fill(0.0)

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
            self.opt.apply(l.params, l.grads)

    def compute_losses(self, y_pred, y_target):
        losses = []
        for i, l in enumerate(self.losses):
            loss = l.compute_loss(y_pred, y_target)
            delta = l.compute_delta(y_pred, y_target)
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

    def compute_metrics(self, y_pred, y_target):
        metrics = []
        for i, m in enumerate(self.metrics):
            me = m.compute_metric(y_pred, y_target)
            metrics.append(me)
        return metrics

    def summary(self, batch_size=1):
        print('==================================')
        print("Model summary")
        print('==================================')

        # Create a dummy input
        ishape = self.l_in[0].oshape
        dummy_x = np.random.random([*ishape, batch_size])

        # Forward
        self.l_in[0].input = dummy_x  # TODO: Temp!
        self.forward()

        for i, l in enumerate(self.fts_layers):
            p_oshape = l.output.shape if l.parent else dummy_x.shape
            print(f"#{i+1}:\t{l.name}\t\t-\t{p_oshape}\t=>\t{l.output.shape}")
