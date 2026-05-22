from operator import index

import numpy as np
import pandas as pd
from fontTools.varLib.models import nonNone

from kmeans import kmeans
from perceptron import Perceptron


class RBFNN(object):
    def __init__(self, numberInput, numberHideUnits, numberOutputs, eta, nameFeatureInputData=None,
                 nameLabelsOutputData=None):
        self.numberInput = numberInput
        self.numberHideUnits = numberHideUnits
        self.numberOutputs = numberOutputs

        if nameFeatureInputData is None:
            self.nameFeatureInputData = ['x{}'.format(index_) for index_ in range(numberInput)]
        else:
            self.nameFeatureInputData = nameFeatureInputData

        if nameLabelsOutputData is None:
            self.nameLabelsOutputData = ['y{}'.format(index_) for index_ in range(numberOutputs)]
        else:
            self.nameLabelsOutputData = nameLabelsOutputData

        self.trainingData = None
        self.trainingDataLabels = None
        self.centorids = None
        self.variances = None
        self.Z = None
        self.eta = eta
        self.perceptron = None
        self.hasbeeentrainedRBFNN = False


    def predict(self,data):
        Y_ = None
        data_ = pd.DataFrame(data,columns=self.nameFeatureInputData)
        if self.hasbeeentrainedRBFNN:
            Z_ = self.calculateZ(data_, self.centorids,self.variances)
            Y_ = self.perceptron.predict(Z_)
        return Y_

    def fit(self, data,dataLabeL,epochs,centroidCalculationProcess=kmeans):
        self.initTrainingData(data,dataLabeL)
        self.fitHideLayer(centroidCalculationProcess)
        self.initPerceptron()
        self.perceptron.fit(self.Z,self.trainingDataLabels['label'].to_numpy(),epochs)
        self.hasbeeentrainedRBFNN =True
    def sumSquared(self, vector1,vector2):
        return np.sum((vector1-vector2)**2)

    def initTrainingData(self, data, dataLabeL):
        self.trainingData = pd.DataFrame(data,columns=self.nameFeatureInputData)
        self.trainingDataLabels = pd.DataFrame(dataLabeL,columns=self.nameLabelsOutputData)

    def fitHideLayer(self, centroidCalculationProcess):
        self.centorids, error_iteration = centroidCalculationProcess(self.numberHideUnits, self.trainingData)
        self.variances = self.calculateVariances(self.trainingData, self.centorids)
        self.Z = self.calculateZ(self.trainingData, self.centorids, self.variances)

    def calculateVariances(self, trainingData, centorids):
        variances_ = list()
        for inddexCentroid_ in range (centorids.shape[0]):
            centorid_ = centorids.iloc[inddexCentroid_]
            filteredData =trainingData[trainingData['centroid'] == inddexCentroid_]
            leastSquare = list()
            for _xIndex in range(filteredData.shape[0]):
                sample = filteredData[self.nameFeatureInputData].iloc[_xIndex]
                leastSquare.append(self.sumSquared(sample,centorid_))
            variances_.append(sum(leastSquare) / filteredData.shape[0])
        return  pd.DataFrame(variances_,columns=['variance'])
    def calculateZ(self, trainingData, centorids, variances):
        Z_ = list()
        for indexSample_ in range(trainingData.shape[0]):
            sample_ = trainingData[self.nameFeatureInputData].iloc[indexSample_]
            row_Z = list()
            for indexCentroid_ in range(centorids.shape[0]):
                centorid_ = centorids[self.nameFeatureInputData].iloc[indexCentroid_]
                variance_ = variances['variance'].iloc[indexCentroid_]
                row_Z.append(np.exp(- self.sumSquared(sample_, centorid_) / (2 * variance_)))
            Z_.append(row_Z)

        return  np.array(Z_)

    def initPerceptron(self):
        self.perceptron =Perceptron(self.eta)

