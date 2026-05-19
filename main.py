import numpy as np
from kmeans import kmeans, generateData
from RNFBR import RBFNN
def testRBFNN():
    data_ = generateData()
    dataT_ = data_[['x','y']]
    dataTLabel_ = dataT_['label']
    rbfnn = RBFNN(2,3,1,1,['x','y'],['label'])
    rbfnn_.fit(dataT_,dataTLabel_,25)
    predictDataY_ = np.round(rbfnn_.predict(dataT_))

    print('...')

