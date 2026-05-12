import numpy
import numpy as np


class Perceptron(object):
    def __init__(self, eta):
        self.eta =eta
        self.W = list()
        self.errorForEachEpoch = list()

    def fit(self, trainingX,trainingY,epochs=50):
        if not self.errorForEachEpoch:
            bestError = np.inf
            W_ = self.initW(trainingX)
        else:
            bestError = min(self.errorForEachEpoch)
            if not self.W:
                W_ = self.initW(trainingX)
            else:
                W_ = self.W_
        for _ in range(epochs):
            pass
    def predict(self, X):
        return self._net_input(X)

    def _net_input(self, X):
        return np.dot(X,self.W[1:]) + self.W[0]
    ##
    def initW(self, trainingX):
        pass


