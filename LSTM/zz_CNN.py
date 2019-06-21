from numpy import genfromtxt
import numpy as np
def prep_data(P,I,DP,PF,DI,U,Target, trainRate=0.8):
    pDat=genfromtxt(P, delimiter=',')
    iDat = genfromtxt(I, delimiter=',')
    dpDat = genfromtxt(DP, delimiter=',')
    pfDat = genfromtxt(PF, delimiter=',')
    diDat = genfromtxt(DI, delimiter=',')
    uDat = genfromtxt(U, delimiter=',')
    tarDat = genfromtxt(Target, delimiter=',')
    row, col = pDat.shape
    trainSize = round(row*trainRate)
    XTrain = np.empty([trainSize, 6, col])
    YTrain = tarDat[0:trainSize, :]
    XTest = np.empty([row-trainSize, 6, col])
    YTest = tarDat[trainSize:, :]
    counterTrain = 0
    counterTest = 0
    for i in range(row):
        if i < trainSize:
            XTrain[counterTrain, 0, :] = pDat[i, :]
            XTrain[counterTrain, 1, :] = iDat[i, :]
            XTrain[counterTrain, 2, :] = dpDat[i, :]/(1500)
            XTrain[counterTrain, 3, :] = pfDat[i, :]
            XTrain[counterTrain, 4, :] = diDat[i, :]/12
            XTrain[counterTrain, 5, :] = uDat[i, :]
            counterTrain += 1
        else:
            XTest[counterTest, 0, :] = pDat[i, :]
            XTest[counterTest, 1, :] = iDat[i, :]
            XTest[counterTest, 2, :] = dpDat[i, :]/(1500)
            XTest[counterTest, 3, :] = pfDat[i, :]
            XTest[counterTest, 4, :] = diDat[i, :]/12
            XTest[counterTest, 5, :] = uDat[i, :]
            counterTest += 1

    return XTrain, YTrain, XTest, YTest