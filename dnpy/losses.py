import numpy as np


class Loss:
    """
    Returns a vector per sample
    """

    def __init__(self, name="Base loss"):
        self.name = name
        self.epsilon = 10e-8

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
        mse = (y_pred-y_target)**2
        loss = float(np.sqrt(np.mean(mse, axis=0, keepdims=True)))
        return loss

    def compute_delta(self, y_pred, y_target):
        mse = (y_pred-y_target)**2  # y_hat and y, are not always positive
        d_loss = (y_pred-y_target)/np.sqrt(mse+self.epsilon)
        return d_loss


class MAE(Loss):

    def __init__(self, name="MAE"):
        super().__init__(name=name)

    def compute_loss(self, y_pred, y_target):
        loss = np.abs(y_pred-y_target)
        loss = float(np.mean(loss, axis=0, keepdims=True))
        return loss

    def compute_delta(self, y_pred, y_target):
        d_loss = np.sign(y_pred-y_target)
        return d_loss


class BinaryCrossEntropy(Loss):

    def __init__(self, name="BinaryCrossEntropy"):
        super().__init__(name=name)

    def compute_loss(self, y_pred, y_target):
        loss = y_target * np.log(y_pred+self.epsilon) + (1.0-y_target) * np.log(1.0-y_pred+self.epsilon)
        loss = -1.0 * float(np.mean(loss, axis=0, keepdims=True))
        return loss

    def compute_delta(self, y_pred, y_target):
        d_loss = y_target * 1.0/(y_pred+self.epsilon) + (1.0-y_target) * 1/(1.0-y_pred+self.epsilon) * -1.0
        d_loss = -1.0 * d_loss
        return d_loss


class CrossEntropy(Loss):

    def __init__(self, name="CrossEntropy"):
        super().__init__(name=name)

    def compute_loss(self, y_pred, y_target):
        # Compute loss: -SUM(p(x) * log q(x_))
        loss = np.sum(y_target.astype(float) * np.log(y_pred+self.epsilon), axis=1, keepdims=True)
        loss = -1.0 * float(np.mean(loss, axis=0, keepdims=True))
        return loss

    def compute_delta(self, y_pred, y_target):
        # d_loss = y_target  # Only valid when the output layer is a softmax
        d_loss = y_target.astype(float) * 1/(y_pred+self.epsilon)
        d_loss = -1.0 * d_loss
        return d_loss


class Hinge(Loss):

    def __init__(self, name="Hinge"):
        super().__init__(name=name)

    def compute_loss(self, y_pred, y_target):
        loss = np.maximum(0, (1-y_pred) * y_target)
        loss = float(np.mean(loss, axis=0, keepdims=True))
        return loss

    def compute_delta(self, y_pred, y_target):
        tmp = (y_pred >= 0).astype(float)
        d_loss = tmp * -1.0 * y_target
        return d_loss
