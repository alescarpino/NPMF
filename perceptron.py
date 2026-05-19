import numpy
import numpy as np
###cambiare nome del file da perceptron e mettere quello che c e scritto nell file edeline???

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
            randomIndices_ = np.array(range(trainingX.shape[0]))
            np.random.shuffle(randomIndices_)
            for iTX, iTY in zip(trainingX[randomIndices_, :],trainingY[randomIndices_]):
                individualUpdate = self.eta * (iTY - self._net_input(iTX,W_))
                W_[1:] += individualUpdate * iTX
                W_[0] += individualUpdate
            error = self.calculateError(trainingX,trainingY,W_)
            self.errorForEachEpoch.append(error)
            if error < bestError:
                bestError =error
                self.W = W_
    def predict(self, X):
        return self._net_input(X, self.W)

    def _net_input(self, X,W):
        return np.dot(X,self.W[1:]) + W[0]
    ##
    def initW(self, trainingX):
        W_ =list()
        totalMean_ = np.mean(trainingX)
        totalStd_ = np.std(trainingX)
        means_ = np.mean(trainingX, axis = 0)
        stds_ = np.std(trainingX, axis= 0)
        random_seed = np.random.RandomState()
        W_.append(random_seed.normal(loc=totalMean_,scale=totalStd_))
        for mean_,std_ in zip(means_,stds_):
            W_.append(random_seed.normal(loc=mean_,scale=std_))

        return np.array(W_)

    def calculateError(self, X, Y, W):
        error_ = .0
        for iX,iY in zip(X, Y,):
            error_ += ((iY - self._net_input(iX,W)) ** 2)
        return error_ /X.shape[0]



