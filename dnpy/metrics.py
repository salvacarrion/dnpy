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

    def compute_metric(self, y_pred, y_target):
        return float(np.mean((y_pred-y_target)**2, axis=0))


class RMSE(Metric):

    def __init__(self, name="MSE"):
        super().__init__(name=name)

    def compute_metric(self, y_pred, y_target):
        return float(np.sqrt(np.mean((y_pred-y_target)**2, axis=1)))


class MAE(Metric):

    def __init__(self, name="MAE"):
        super().__init__(name=name)

    def compute_metric(self, y_pred, y_target):
        return float(np.mean(np.abs(y_pred-y_target), axis=0))


class BinaryAccuracy(Metric):

    def __init__(self, name="BinaryAccuracy"):
        super().__init__(name=name)

    def compute_metric(self, y_pred, y_target):
        # Get classes
        y_target_best_class = y_target.astype(bool)
        y_pred_best_class = y_pred > 0.5

        # Check class correctness
        is_correct = y_target_best_class == y_pred_best_class

        # Compute accuracy
        acc = float(np.mean(is_correct.astype(int), axis=1))
        return acc


class CategoricalAccuracy(Metric):

    def __init__(self, name="CategoricalAccuracy"):
        super().__init__(name=name)

    def compute_metric(self, y_pred, y_target):
        # Get maximum values (aka. predicted class)
        y_target_best_class = np.argmax(y_target, axis=1)
        y_pred_best_class = np.argmax(y_pred, axis=1)

        # Check class correctness
        is_correct = y_target_best_class == y_pred_best_class

        # Compute accuracy
        acc = float(np.mean(is_correct.astype(int)))
        return acc
