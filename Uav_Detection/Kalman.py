import cv2 as cv
import numpy as np

kf = cv.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)

def Tahmin(X, Y):

    olcum = np.array([[np.float32(X)], [np.float32(Y)]])
    kf.correct(olcum)
    onerilen = kf.predict()

    return onerilen

