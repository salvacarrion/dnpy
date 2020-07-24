from dnpy import losses
import numpy as np


class Metric:

    def __init__(self, name="Base loss"):
        self.name = name

    def compute_metric(self, y_pred, y_target):
        pass


class MSE(Metric):

    def __init__(self, name="MSE"):
        super().__init__(name=name)
        self.cls = losses.MSE()

    def compute_metric(self, y_pred, y_target):
        return self.cls.compute_loss(y_pred, y_target)


class RMSE(Metric):

    def __init__(self, name="MSE"):
        super().__init__(name=name)
        self.cls = losses.RMSE()

    def compute_metric(self, y_pred, y_target):
        return self.cls.compute_loss(y_pred, y_target)


class MAE(Metric):

    def __init__(self, name="MAE"):
        super().__init__(name=name)
        self.cls = losses.MAE()

    def compute_metric(self, y_pred, y_target):
        return self.cls.compute_loss(y_pred, y_target)


class BinaryAccuracy(Metric):

    def __init__(self, name="BinaryAccuracy"):
        super().__init__(name=name)

    def compute_metric(self, y_pred, y_target):
        # Get classes
        y_target_class = y_target.astype(bool)
        y_pred_class = (y_pred > 0.5).astype(bool)

        # Check class correctness
        is_correct = y_target_class == y_pred_class

        # Compute accuracy
        acc = float(np.mean(is_correct, axis=1))
        return acc


class CategoricalAccuracy(Metric):

    def __init__(self, name="CategoricalAccuracy"):
        super().__init__(name=name)

    def compute_metric(self, y_pred, y_target):
        y_pred, y_target = y_pred.T, y_target.T

        # Get maximum values (aka. predicted class)
        y_target_best_class = np.argmax(y_target, axis=1)
        y_pred_best_class = np.argmax(y_pred, axis=1)

        # Check class correctness
        is_correct = y_target_best_class == y_pred_best_class

        # Compute accuracy
        acc = float(np.mean(is_correct.astype(int)))
        return acc
