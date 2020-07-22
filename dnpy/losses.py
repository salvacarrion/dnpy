import numpy as np


class Loss:
    """
    Returns a vector per sample
    """

    def __init__(self, name="Base loss"):
        self.name = name

    def compute_loss(self, y_pred, y_target):
        pass


class MSE(Loss):

    def __init__(self, name="MSE"):
        super().__init__(name=name)

    def compute_loss(self, y_pred, y_target):
        return float(np.mean((y_pred-y_target)**2, axis=0, keepdims=True))

    def compute_delta(self, y_pred, y_target):
        return 2 * (y_pred-y_target)


class RMSE(Loss):

    def __init__(self, name="RMSE"):
        super().__init__(name=name)

    def compute_loss(self, y_pred, y_target):
        return float(np.sqrt(np.mean((y_pred-y_target)**2, axis=1, keepdims=True)))

    def compute_delta(self, y_pred, y_target):
        mse = (y_pred-y_target)**2
        return 1/(2*np.sqrt(mse)) * 2*(y_pred-y_target)


class MAE(Loss):

    def __init__(self, name="MAE"):
        super().__init__(name=name)

    def compute_loss(self, y_pred, y_target):
        return float(np.mean(np.abs(y_pred-y_target), axis=1, keepdims=True))

    def compute_delta(self, y_pred, y_target):
        return np.sign(y_pred-y_target)


class BinaryCrossEntropy(Loss):

    def __init__(self, name="BinaryCrossEntropy"):
        super().__init__(name=name)

    def compute_loss(self, y_pred, y_target):
        loss = y_target * np.log10(y_pred) + (1-y_target) * np.log10(1-y_pred)
        loss = -1.0 * np.sum(loss, axis=0, keepdims=True)
        return float(np.mean(loss))

    def compute_delta(self, y_pred, y_target):
        d_loss = y_target * 1/y_pred + (1 - y_target) * 1/(1 - y_pred)
        d_loss = -1.0 * d_loss
        return d_loss


class CrossEntropy(Loss):

    def __init__(self, name="CrossEntropy", num_classes=2):
        super().__init__(name=name)
        self.num_classes = num_classes

    def compute_loss(self, y_pred, y_target):
        # Get maximum values (aka. predicted class)
        y_target_best_class = np.argmax(y_target, axis=1)
        y_pred_best_class = np.argmax(y_pred, axis=1)

        # Get probabilities of the predicted class
        y_target_best_prob = y_pred[np.arange(0, len(y_pred)), y_target_best_class]

        # Check class correctness
        is_correct = y_target_best_class == y_pred_best_class

        # Compute loss
        loss = is_correct.astype(float) * np.log(y_target_best_prob)
        return float(np.mean(loss))

    def compute_delta(self, y_pred, y_target):
        # Get maximum values (aka. predicted class)
        y_target_best_class = np.argmax(y_target, axis=1)
        y_pred_best_class = np.argmax(y_pred, axis=1)

        # Get probabilities of the predicted class
        y_target_best_prob = y_pred[np.arange(0, len(y_pred)), y_target_best_class]

        # Check class correctness
        is_correct = y_target_best_class == y_pred_best_class

        # Compute loss
        d_loss = is_correct.astype(float) * 1/y_target_best_prob
        return d_loss
