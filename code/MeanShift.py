import random
from scipy.spatial import distance
import numpy as np

def MeanShift(X, h):
    nDims = X.shape[1]
    
    finalPoints = np.zeros((X.shape[0], nDims))

    for pos, point in enumerate(X):
        point = point.reshape(1, -1)

        notconvergence = True

        prePoint = 0
        while(notconvergence):

            dist = distance.cdist(X, point, 'euclidean')
            #print(dist)
            insideWindow = np.where(dist < h, X, 0)
            b = insideWindow.shape[0]
            insideWindow = insideWindow[insideWindow != 0]
            inS = (int)(insideWindow.shape[0] / nDims)
            insideWindow = insideWindow[0:inS*nDims]
            insideWindow = insideWindow.reshape(inS, nDims)

            pointsMean = np.mean(insideWindow, axis=0)
            if len(pointsMean.shape) == 1:
                pointsMean = pointsMean.reshape(1, -1)

            point = pointsMean
            #print(pointsMean)

            if (prePoint == point).all():
                notconvergence = False
            else:
                prePoint = point

        finalPoints[pos] = point
        
    return finalPoints