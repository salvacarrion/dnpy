import numpy as np


class Regularizer:

    def __init__(self, name="Base regularizer"):
        self.name = name

    def forward(self, param):
        pass

    def backward(self, param):
        pass


class L1(Regularizer):

    def __init__(self, lmda=0.01):
        super().__init__(name="L1")
        self.lmda_l1 = lmda

    def forward(self, param):
        return self.lmda_l1 * np.abs(param)

    def backward(self, param):
        return self.lmda_l1 * np.sign(param)


class L2(Regularizer):

    def __init__(self, lmda=0.01):
        super().__init__(name="L2")
        self.lmda_l2 = lmda

    def forward(self, param):
        return self.lmda_l2 * (param**2)

    def backward(self, param):
        return self.lmda_l2 * (2 * param)


class L1L2(Regularizer):

    def __init__(self, lmda_l1=0.01, lmda_l2=0.01):
        super().__init__(name="L1L2")
        self.l1_reg = L1(lmda_l1)
        self.l2_reg = L2(lmda_l2)

    def forward(self, param):
        return self.l1_reg.forward(param) + self.l2_reg.forward(param)

    def backward(self, param):
        return self.l1_reg.backward(param) + self.l2_reg.backward(param)
