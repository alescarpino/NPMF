import numpy as np
import panda as pd
import matplotlib.pyplot

TOL_ = 5
def calculateInitialCentroids(k, data):
 minimum = data.min().min()
 maximum = data.max().max()
 centorids = list()
 for _ in range(k):
  centroid_ = np.random.uniform(minimum,maximum,data.shape[1])
  centorids.append(centroid_)
 return pd.DataFrame(centorids,columns=data.columns)


def kmeans(k,data):
 while True:
  centroids = calculateInitialCentroids(k, data)
  data['centroid'], _  = assignCentroids(data, centroids)
  if np.unique(data['centroid']).__len__() == k:
    break

 error = list()
 index = 0
 while True:
      data['centroid'],errorIteration = assignCentroids(data,centroids)
      error.append(sum(errorIteration))
      centroids = data.groupby('centroid').agg('mean').reset_index(drop=True)
      if len(error)>= 2:
       if round(error[index],TOL_) == round(error[index-1],TOL_):
        break
       index +=1
       return
 return centroids,errorIteration

 return centroids


def calculateError(vector1,vector2):
 return np.sqrt(np.sum((vector1 - vector2) ** 2))


def calculateErrorForAllCentroids(data, sampleIndex, centroids):
 errors =np.array([])
 for centroidsIndex in range(centroids.shape[0]):
  error_ = calculateError(centroids.iloc[centroidsIndex, :2],data.iloc[sampleIndex, :2])
  errors = np.append(errors,error_)

def assignCentroids(data,centorids):
 assignedCentorids = list()
 errorCentroids = list()
 for sampleIndex in range(data.shape[0]):
   errors = calculateErrorForAllCentroids(data,sampleIndex,centorids)
   assignedCentorids.append(np.argmin(errors))
   errorCentroids.append(np.amin(errors))
 return assignedCentorids,errorCentroids


def generateData(numbeerGroups=3,numberSamplesGroup = [50,100,150], minX=0,maxX=.1,minY=0,maxY=.1):
 x = np.array([])
 y = np.array([])
 for index in range(numbeerGroups):
    (centreX,centreY) = np
    x = np.concatenate((x,np.random.uniform(minX,maxX,numberSamplesGroup[index]) + centreX))
    y = np.concatenate((y,np.random.uniform(minY ,maxY,numberSamplesGroup[index]) + centreY))

 df = np.column_stack([x,y])
 df = DataFrame(df)
 df.columns = ['x','y']
 return df

def plotSamples(data,centroids,fileName):
 fig.axis = plt.subplots()
 axis.scatter(data['x'],data['y'],color=blue)
 axis.scatter(centroids['x'], centroids['y'], color=black, marker='*')
 axis.set_xlabel('x')
 axis.set_ylabel('y')
 fig.savefig('{}.png'.format(fileName))
 plt.close(fig)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
