import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as KB
import os 
from math import log
from keras.layers import Dense, Dropout
import keras
import math
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import _num_samples
import pickle

class AdaBoost():
    def __init__(self, n_est, loss, learning_rate, kerasEpochs, kerasBatchSize):
        self.n_est = n_est
        self.loss = loss
        self.estimators = []
        self.estimatorsErrors = []
        self.estimatorsWeights = []
        self.estimatorsSampleWeights = []
        self.learning_rate = learning_rate
        self.averageLoss = []
        self.kerasEpochs = kerasEpochs
        self.kerasBatchSize = kerasBatchSize
        
        
    #Loss function  for Keras    
    def deviance(self, data, y_pred):
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
    
    def devSingle(self, y, yhat, d):
        if y != 0:
            return 2 * d * (y * log(y) - y * log(yhat) - y + yhat)
        else:
            return 2 * d * yhat
        return print("error")
    
    def devFull(self, y, yhat, d):
        sumtot = 0
        y = pd.DataFrame(y)
        yhat = pd.DataFrame(yhat)
        d = pd.DataFrame(d)
        arr = np.append(y, yhat, axis = 1)
        arr = np.append(arr, d, axis = 1)
        arr = pd.DataFrame(arr)
        arr.columns = ["y", "yhat", "d"]
        for index, row in arr.iterrows():
            dev = self.devSingle(row["y"], row["yhat"], row["d"])
            sumtot = sumtot + dev
        return sumtot
    
    #MODEL USED FOR BOOSTING
    def baseline_model2(self, dropout = 0.2, kernel_initializer = 'uniform', nn1 = 5, lr = 0.15, act1 = "softmax"):
        with tf.device('/gpu:0'):
            # create model
            #building model
            model = keras.Sequential()
            model.add(Dense(nn1, input_dim = 21, activation = act1, kernel_initializer=kernel_initializer))
            model.add(Dropout(dropout))
            #model.add(Dense(2, activation = "exponential"))
            model.add(Dense(1, activation = "exponential", kernel_initializer=kernel_initializer))
            optimizer = keras.optimizers.adagrad(lr=lr)
            model.compile(loss=self.deviance, optimizer=optimizer, metrics = [self.deviance, "mean_squared_error"])
            return model
        
    #MODEL USED FOR BOOSTING
    def baseline_model(self, dropout = 0.2, kernel_initializer = 'uniform', nn1 = 5, lr = 0.15, act1 = "softmax"):
        with tf.device('/gpu:0'):
            # create model
            #building model
            model = keras.Sequential()
            model.add(Dense(nn1, input_dim = 21, activation = act1, kernel_initializer=kernel_initializer))
            model.add(Dropout(dropout))
            #model.add(Dense(2, activation = "exponential"))
            model.add(Dense(1, activation = "exponential", kernel_initializer=kernel_initializer))
            optimizer = keras.optimizers.adagrad(lr=lr)
            model.compile(loss=self.deviance, optimizer=optimizer, metrics = [self.deviance, "mean_squared_error"])
            return model
        
    #inits weights to 1/N
    def initWeights(self, data):
        n = len(data)
        weights = np.ones(n)
        weights = weights/sum(weights)
        return weights 
        
    
    #Normalizing the weights to sum = 1
    def normalizeWeights(self, weights):
        sumW = sum(weights)
        newW = weights/sumW
        return newW
    
    #Resample data given weights, returns indices of said new data.
    def resampleIndices(self, data, weights):
        nRows = len(data)
        indices = np.arange(nRows)
        res = np.random.choice(indices, size = nRows, replace = True, p = weights)
        return res
    
    #Selecting rows with array of indices
    def dataFromIndices(self, data, indices):
        newD = data.iloc[indices, :]
        return newD
    
    def printeq(self):
        print('==================================================')
    
    
    #The reason feed is passed instead of Y is because keras needs a tuple (y, d) to calculate the custom loss function deviance.
    #data is X
    def boost(self, iboost, data, feed, weights):
        y = feed.iloc[:,0]
        d = feed.iloc[:,1]
        
        #setting up the estimator
        #estimator = self.baseline_model2()
        estimator = KerasRegressor(build_fn=self.baseline_model)
        
        #weighted sampling
        weightedSampleIndices = self.resampleIndices(data, weights)
        feedSampled = self.dataFromIndices(feed, weightedSampleIndices)
        dataSampled = self.dataFromIndices(data, weightedSampleIndices)
        
        #fit on boostrapped sample
        estimator.fit(dataSampled, feedSampled, batch_size=self.kerasBatchSize, epochs = self.kerasEpochs, verbose=2)
        self.estimators.append(estimator)
            
        

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
        if self.loss == 'square':
            masked_error_vector **= 2
        elif self.loss == 'exponential':
            masked_error_vector = 1. - np.exp(-masked_error_vector)
            
        #average loss
        estimator_error = (masked_sample_weight * masked_error_vector).sum()
    
        #TODO STOP IF ESTIMATOR ERROR <= 0
        
        if estimator_error >= 0.5:
            #discard current if it is not first one
            if(len(self.estimators) > 1):
                self.estimators.pop(-1)
                self.printeq()
                print("Average Loss >= 0.5, stopping AdaBoost.R2 after ", iboost, " runs.")
                self.printeq()
            return None, None, None, None
        
        
        #beta
        beta = estimator_error / (1. - estimator_error)
        estimator_weight = self.learning_rate * np.log(1. / beta)
        
        if not iboost == self.n_est - 1:
            weights[sample_mask] *= np.power(
                beta, (1.-masked_error_vector) * self.learning_rate
            )
            
        #append self variables TODO + ADD SAMPLES USED FOR TRAIN
        
        return weights, estimator_weight, estimator_error, estimator
    
    def fit(self, data, feed, weights):
        #check if adaboost loss is correctly setup
        if self.loss not in ('linear', 'square', 'exponential'):
            raise ValueError(
                "loss must be 'linear', 'square', or 'exponential'")
        
        #normalize weights 
        weights = self.normalizeWeights(weights)
        
        #clearing vars
        self.estimators = []
        self.estimatorsWeights = np.zeros(self.n_est, dtype=np.float64)

        self.estimatorsSampleWeights = []
        #append first one
        self.estimatorsSampleWeights.append(weights)
        self.estimatorsErrors = np.ones(self.n_est, dtype=np.float64)        
        #looping
        self.printeq()
        print("AdaBoost.R2, Lucien Ledune Masters Thesis")
        self.printeq()
        print(self.n_est, " estimators will be fit (at most)")
        for iboost in range(self.n_est):
            self.printeq()
            print("Fitting estimator number ", iboost + 1)
            
            weights, estimator_weight, estimator_error, estimator = self.boost(iboost, data, feed, weights)
            
            #check if end early
            if weights is None:
                break
            
            #normalize weights
            weights = self.normalizeWeights(weights)
            
            #append        
            self.estimatorsWeights[iboost] = estimator_weight
            self.estimatorsErrors[iboost] = estimator_error
            self.estimatorsSampleWeights.append(weights)            
            self.printeq()
            print("Done.")
            self.printeq()
            
            if(estimator_error == 0):
                break
            
            weights_sum = np.sum(weights)
            
            #stop is weights are negative
            if(weights_sum <= 0):
                break
            
        return self
    
    def getMedianPred(self, data, limit):
        predictions = np.array([est.predict(data) for est in self.estimators[:limit]]).T
        sorted_idx = np.argsort(predictions, axis = 1)
        #find index of median prediction for each sample
        weight_cdf = stable_cumsum(self.estimatorsWeights[sorted_idx], axis = 1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis = 1)
        median_estimators = sorted_idx[np.arange(_num_samples(data)), median_idx]
        
        #return median preds
        return predictions[np.arange(_num_samples(data)), median_estimators]
    
    def predict(self, data):
        return self.getMedianPred(data, len(self.estimators))
    
    def stagedPredict(self, data):
        for i, _ in enumerate(self.estimators, 1):
            yield self.getMedianPred(data, limit = i)
            
    def save_model(self, path):
        #config dictionary and weights
        config_dic = {
                'n_est' : self.n_est,
                'loss' : self.loss,
                'estimatorsErrors' : self.estimatorsErrors,
                'estimatorsWeights' : self.estimatorsWeights,
                'estimatorsSampleWeights' : self.estimatorsSampleWeights,
                'learning_rate' : self.learning_rate,
                'averageLoss' : self.averageLoss,
                'kerasEpochs' : self.kerasEpochs,
                'kerasBatchSize' : self.kerasBatchSize
            }
        dictpath = path + '/savedDict'
        with open(dictpath, 'wb') as dictfile:
            pickle.dump(config_dic, dictfile)
        
        #models
        models = self.estimators
        pathModelFolder = path + '/models'
        counter = 0
        for model in models:
            #convert from keras regressor to keras model 
            modelKeras = model.model
            #save it 
            counter = counter + 1
            singleModelPath = pathModelFolder + "/model" + str(counter)
            modelKeras.save(singleModelPath)
        return
    
    #pathADB is the path to AdaBoost folder (same as used in save_model) 
    def load_model(self, pathADB):
        print('Loading model ...')
        dicpath = pathADB + '/savedDict'
        with open(dicpath, 'rb') as configDicFile:
            config_dic = pickle.load(configDicFile)
        #assign configs
        self.n_est = config_dic['n_est']
        self.loss = config_dic['loss']
        self.estimatorsErrors = config_dic['estimatorsErrors']
        self.estimatorsWeights = config_dic['estimatorsWeights']
        self.estimatorsSampleWeights = config_dic['estimatorsSampleWeights']
        self.learning_rate = config_dic['learning_rate']
        self.averageLoss = config_dic['averageLoss']
        
        #models
        estLoaded = []
        modelsPath = pathADB + '/models'
        for i in range(0, len(self.estimatorsWeights)):
            modelString = '/model' + str(i+1)
            modelFullString = modelsPath + modelString
            
            model = KerasRegressor(build_fn=self.baseline_model2, epochs = 300, batch_size = 10, verbose = 2)
            model.model = keras.models.load_model(modelFullString, custom_objects={'deviance' : self.deviance})
            estLoaded.append(model)
        self.estimators = estLoaded
        
        print("Model loaded.")
            
        
        
#importing datasets
data = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/preprocFull.csv")

#separing columns
d1 = data['Duration']
y1 = data['ClaimFrequency']
nc1 = data['NumberClaims']

#dropping useless dimensions
data = data.drop(columns=["Duration", "NumberClaims", "ClaimFrequency"])

#Passing the Duration into keras is impossible cause there is two arguments only when creating a custom loss function.
#Therefore we use a trick and pass a tuple with duration and y instead. 
y1 = pd.DataFrame(y1)
d1 = pd.DataFrame(d1)
feed = np.append(y1, d1, axis = 1)
feed = pd.DataFrame(feed)

est = AdaBoost(2, 'linear', learning_rate = 1, kerasBatchSize=5000, kerasEpochs=50)

initWeights = est.initWeights(data)

est.fit(data, feed, initWeights)

vectorWeights = est.estimatorsWeights
vectorSampleWeights = est.estimatorsSampleWeights
vectorEstimators = est.estimators
vectorErrors = est.estimatorsErrors
            
xxx = est.predict(data)
devTest = est.devFull(y1, xxx, d1)

pathtosave = "c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/python/models/AdaBoost"
est.save_model(pathtosave)

##########
#test load
adatest = AdaBoost(2, 'linear', 1, kerasBatchSize=5000, kerasEpochs=50)
adatest.load_model(pathtosave)
xxxx = adatest.predict(data)
xxxxdevTest = adatest.devFull(y1, xxxx, d1)




