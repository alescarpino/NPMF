import numpy as np
from kmeans import kmeans, generateData
from RNFBR import RBFNN


def testRBFNN():
    data_ = generateData()
    dataT_ = data_[['x', 'y']]
    dataTLabel_ = dataT_['label']

    # Manca la creazione dell'oggetto corretto
    rbfnn = RBFNN(2, 3, 1, 0.01, ['x', 'y'], ['label'])  # eta (learning rate) impostato a 0.01
    rbfnn.fit(dataT_, dataTLabel_, 25)
    predictDataY_ = np.round(rbfnn.predict(dataT_))

    print('Predizioni:', predictDataY_)

