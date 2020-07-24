import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as KB
import os 
from math import log
from keras.layers import Dense, Dropout
import keras

#importing datasets
data = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/preprocFull.csv")
datacats = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/catOnly.csv")

#separing columns
d1 = data['Duration']
d2 = datacats['Duration']

y1 = data['ClaimFrequency']
y2 = datacats['NumberClaims']/datacats['Duration']

nc1 = data['NumberClaims']
nc2 = datacats['NumberClaims']

#dropping useless dimensions
data = data.drop(columns=["Duration", "NumberClaims", "ClaimFrequency"])
datacats = datacats.drop(columns=["Unnamed: 0", "Duration", "NumberClaims", "ClaimCost"])

#Passing the Duration into keras is impossible cause there is two arguments only when creating a custom loss function.
#Therefore we use a trick and pass a tuple with duration and y instead. 
y1 = pd.DataFrame(y1)
y2 = pd.DataFrame(y2)
d1 = pd.DataFrame(d1)
d2 = pd.DataFrame(d2)
feed = np.append(y1, d1, axis = 1)
feed2 = np.append(y2, d2, axis = 1)

feed = pd.DataFrame(feed)
feed2 = pd.DataFrame(feed2)

#Loss function     
def deviance(data, y_pred):
        y_true = data[:, 0]
        d = data[:, 1]
        
        lnY = KB.log(y_true)
        bool1 = KB.equal(y_true, 0)
        zeros = KB.zeros_like(y_true)
        lnY = KB.switch(bool1, zeros, lnY)
        
        lnYp = KB.log(y_pred)
        bool2 = KB.equal(y_pred, 0)
        zeross = KB.zeros_like(y_pred)
        lnYp = KB.switch(bool2, zeross, lnYp)
        
        loss = 2 * d * (y_true * lnY - y_true * lnYp[:, 0] - y_true + y_pred[:, 0])
        return loss

def devSingle(y, yhat, d):
    if y != 0:
        return 2 * d * (y * log(y) - y * log(yhat) - y + yhat)
    else:
        return 2 * d * yhat
    return print("error")

def devFull(y, yhat, d):
    sumtot = 0
    y = pd.DataFrame(y)
    yhat = pd.DataFrame(yhat)
    d = pd.DataFrame(d)
    arr = np.append(y, yhat, axis = 1)
    arr = np.append(arr, d, axis = 1)
    arr = pd.DataFrame(arr)
    arr.columns = ["y", "yhat", "d"]
    for index, row in arr.iterrows():
        dev = devSingle(row["y"], row["yhat"], row["d"])
        sumtot = sumtot + dev
    return sumtot

#MODEL USED FOR BOOSTING
def baseline_model2(dropout = 0.2, kernel_initializer = 'uniform', nn1 = 15, lr = 0.01, act1 = "softmax"):
    with tf.device('/gpu:0'):
        # create model
        #building model
        model = keras.Sequential()
        model.add(Dense(nn1, input_dim = 21, activation = act1, kernel_initializer=kernel_initializer))
        model.add(Dropout(dropout))
        #model.add(Dense(2, activation = "exponential"))
        model.add(Dense(1, activation = "exponential", kernel_initializer=kernel_initializer))
        optimizer = keras.optimizers.adagrad(lr=lr)
        model.compile(loss=deviance, optimizer=optimizer, metrics = [deviance, "mean_squared_error"])
        return model


##############################################
#ADABOOST.R2
##############################################


#inits weights to 1/N
def initWeights(data):
    n = len(data)
    weights = np.ones(n)
    weights = weights/sum(weights)
    return weights 
    

#Normalizing the weights to sum = 1
def normalizeWeights(weights):
    sumW = sum(weights)
    newW = weights/sumW
    return newW

#Resample data given weights, returns indices of said new data.
def resampleIndices(data, weights):
    nRows = len(data)
    indices = np.arange(nRows)
    res = np.random.choice(indices, size = nRows, replace = True, p = weights)
    return res

#Selecting rows with array of indices
def dataFromIndices(data, indices):
    newD = data.iloc[indices, :]
    return newD

aaa = np.zeros(len(data))
aaa[0] = 1
aaa[1] = 2
weights = normalizeWeights(aaa)

def discardEstimator(estArray, currentEstError):
    if(currentEstError >= 0.5)

#The reason feed is passed instead of Y is because keras needs a tuple (y, d) to calculate the custom loss function deviance.
#data is X
def boost(iboost, data, feed, weights, loss = 'linear'):
    y = feed.iloc[:,0]
    d = feed.iloc[:,1]
    
    #setting up the estimator
    estimator = baseline_model2()
    
    #weighted sampling
    weightedSampleIndices = resampleIndices(data, weights)
    feedSampled = dataFromIndices(feed, weightedSampleIndices)
    dataSampled = dataFromIndices(data, weightedSampleIndices)
    
    #fit on boostrapped sample
    estimator.fit(dataSampled, feedSampled, batch_size=5000, epochs = 20, verbose=2)
    #get estimates on initial dataset
    preds = pd.DataFrame(estimator.predict(data)).iloc[:,0]
    
    #error vector
    error_vect = y-preds
    error_vect = np.abs(error_vect)    
    sample_mask = weights > 0
    masked_sample_weight = weights[sample_mask]
    masked_error_vector = error_vect[sample_mask]
    
    #max error
    error_max = masked_error_vector.max()
    if error_max != 0:
        #normalizing
        masked_error_vector /= error_max
    #if loss isn't linear then modify it accordingly
    if loss == 'square':
        masked_error_vector **= 2
    elif loss == 'exponential':
        masked_error_vector = 1. - np.exp(-masked_error_vector)
        
    #average loss
    estimator_error = (masked_sample_weight * masked_error_vector).sum()

    #TODO STOP IF ESTIMATOR ERROR <= 0

    elif        
    
        
    
    
    

#Boost 























