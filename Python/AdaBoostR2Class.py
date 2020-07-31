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
import time

#########################################
#!!! change root to your main folder !!
#########################################
root = 'c:/users/kryst/desktop/poisson/poisson-neural-network-insurance-pricing'




class AdaBoost():
    """
    A classed used for AdaBoost implementation
    
    ...
    
    Attributes
    ----------
    n_est : int
    loss : [exponential, linear, square]
    learning_rate : int > 0
    kerasEpochs : int
    kerasBatchSize : int
    dropout : [0,1]
    nn1 : int
    keraslr : [0,1]
    input_dim : int
    
    """
    
    
    def __init__(self, n_est = 50, loss = 'exponential', learning_rate = 1, kerasEpochs = 300, 
                 kerasBatchSize = 1000, dropout = 0.2, nn1 = 5, keraslr = 0.1, input_dim = 21):
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
        self.dropout = dropout
        self.nn1 = nn1
        self.keraslr = keraslr
        self.input_dim = input_dim
        
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
    
    def devArray(self, y, yhat, d):
        """
        returns array of all deviance values
        """
        n = len(y)
        dev = np.ndarray(n)
        y = pd.DataFrame(y)
        yhat = pd.DataFrame(yhat)
        d = pd.DataFrame(d)
        y = y.values
        yhat = yhat.values
        d = d.values

        #convert params to ndarray
        for i in range(0, n):
            temp = self.devSingle(y[i], yhat[i], d[i])
            dev[i] = temp
        return dev
        
        
    def devFull(self, y, yhat, d):
        """
        Returns full deviance

        Parameters
        ----------
        y : vector
            targets.
        yhat : vector
            predictions.
        d : vector
            duration.

        Returns
        -------
        sumtot : int
            Full deviance (sum)

        """
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
    def baseline_model(self, kernel_initializer = 'uniform', act1 = "softmax"):
        with tf.device('/gpu:0'):
            # create model
            #building model
            model = keras.Sequential()
            model.add(Dense(self.nn1, input_dim = self.input_dim, activation = act1, kernel_initializer=kernel_initializer))
            model.add(Dropout(self.dropout))
            #model.add(Dense(2, activation = "exponential"))
            model.add(Dense(1, activation = "exponential", kernel_initializer=kernel_initializer))
            optimizer = keras.optimizers.adagrad(lr=self.keraslr)
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
        y = feed.iloc[:,0].values
        d = feed.iloc[:,1].values
        
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
        
        if(self.loss != 'deviance'):
            #error vector
            #the error vector has length 100??
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
        else:
            #if we use deviance
            error_vect = self.devArray(y, preds, d)
            sample_mask = weights > 0
            masked_sample_weight = weights[sample_mask]
            masked_error_vector = error_vect[sample_mask]
            #max error
            error_max = masked_error_vector.max()
            if error_max != 0:
                masked_error_vector/= error_max
            for i in range(0,100):
                print(masked_error_vector.max())
                
            #average loss
            estimator_error = (masked_sample_weight * masked_error_vector).sum()
            
        print('Average loss : ', estimator_error)
    
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
        """
        Fit function for a model

        Parameters
        ----------
        data : data
        feed : feed
        weights : weights

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        #check if adaboost loss is correctly setup
        if self.loss not in ('linear', 'square', 'exponential', 'deviance'):
            raise ValueError(
                "loss must be 'linear', 'square', or 'exponential', or 'deviance'")
        
        
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
            print("Fitting estimator number ", iboost + 1, " on ", self.n_est)
            
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
                'kerasBatchSize' : self.kerasBatchSize,
                'dropout' : self.dropout,
                'nn1' : self.nn1,
                'keraslr' : self.keraslr,
                'input_dim' : self.input_dim
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
        self.kerasEpochs = config_dic['kerasEpochs']
        self.kerasBatchSize = config_dic['kerasBatchSize']
        self.dropout = config_dic['dropout']
        self.nn1 = config_dic['nn1']
        self.keraslr = config_dic['keraslr']
        self.input_dim = config_dic['input_dim']
        
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
dataTrain = pd.read_csv(root + "/dataTrain.csv")
datacatsTrain = pd.read_csv(root + "/dataCatsTrain.csv")

#separing columns
d1 = dataTrain['Duration']
d2 = datacatsTrain['Duration']

y1 = dataTrain['ClaimFrequency']
y2 = datacatsTrain['NumberClaims']/datacatsTrain['Duration']

nc1 = dataTrain['NumberClaims']
nc2 = datacatsTrain['NumberClaims']

#importing test 
dataTest = pd.read_csv(root + "/dataTest.csv")
datacatsTest = pd.read_csv(root + "/dataCatsTest.csv")

d1test = dataTest['Duration']
d2test = datacatsTest['Duration']

y1test = dataTest['ClaimFrequency']
y2test = datacatsTest['NumberClaims']/datacatsTest['Duration']

#dropping useless dimensions
dataTrain = dataTrain.drop(columns=["Duration", "NumberClaims", "ClaimFrequency", "Unnamed: 0"])
datacatsTrain = datacatsTrain.drop(columns=["Unnamed: 0", "Duration", "NumberClaims", "ClaimCost", "Unnamed: 0"])
dataTest = dataTest.drop(columns=["Duration", "NumberClaims", "ClaimFrequency", "Unnamed: 0"])
datacatsTest = datacatsTest.drop(columns=["Unnamed: 0", "Duration", "NumberClaims", "ClaimCost", "Unnamed: 0"])

#Passing the Duration into keras is impossible cause there is two arguments only when creating a custom loss function.
#Therefore we use a trick and pass a tuple with duration and y instead. 
y1 = pd.DataFrame(y1)
y2 = pd.DataFrame(y2)
d1 = pd.DataFrame(d1)
d2 = pd.DataFrame(d2)

y1test = pd.DataFrame(y1test)
y2test = pd.DataFrame(y2test)
d1test = pd.DataFrame(d1test)
d2test = pd.DataFrame(d2test)

feed = np.append(y1, d1, axis = 1)
feed2 = np.append(y2, d2, axis = 1)
feed = pd.DataFrame(feed)
feed2 = pd.DataFrame(feed2)

'''
est = AdaBoost(10, 'linear', learning_rate = 1, kerasBatchSize=51600, kerasEpochs=150)

initWeights = est.initWeights(dataTrain)

est.fit(dataTrain, feed, initWeights)

vectorWeights = est.estimatorsWeights
vectorSampleWeights = est.estimatorsSampleWeights
vectorEstimators = est.estimators
vectorErrors = est.estimatorsErrors
            
xxx = est.predict(dataTest)
devTest = est.devFull(y1test, xxx, d1test)

pathtosave = "c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/python/models/AdaBoost"
est.save_model(pathtosave)

##########
#test load
adatest = AdaBoost(2, 'linear', 1, kerasBatchSize=5000, kerasEpochs=50)
adatest.load_model(pathtosave)
xxxx = adatest.predict(data)
xxxxdevTest = adatest.devFull(y1, xxxx, d1)

'''
#####################################
# PARAMETER SEARCH ##################
#####################################
#The results are stored in lists
#####################################

import random
#Defining param grid 
param_grid = {
        'n_est' : [100],
        'loss' : ['deviance'],
        'learning_rate' : [1],
        'kerasEpochs' : [100],
        'kerasBatchSize' : [51600],
        'dropout' : [0.1,0.2,0.3],
        'nn1' : [5,10,15],
        'keraslr' : [0.1],
    }


paramdraws = []
nTests = 1

print(nTests, " tests will be done.")

#generate all the grids for testing 
for i in range(0, nTests): 
    paramdraw = {
            'n_est' : [],
            'loss' : [],
            'learning_rate' : [],
            'kerasEpochs' : [],
            'kerasBatchSize' : [],
            'dropout' : [],
            'nn1' : [],
            'keraslr' : [],
        }
    #choosing randomly
    paramdraw['n_est'].append(random.choice(param_grid['n_est']))
    paramdraw['loss'].append(random.choice(param_grid['loss']))
    paramdraw['learning_rate'].append(random.choice(param_grid['learning_rate']))
    paramdraw['kerasEpochs'].append(random.choice(param_grid['kerasEpochs']))
    paramdraw['kerasBatchSize'].append(random.choice(param_grid['kerasBatchSize']))
    paramdraw['dropout'].append(random.choice(param_grid['dropout']))
    paramdraw['nn1'].append(random.choice(param_grid['nn1']))
    paramdraw['keraslr'].append(random.choice(param_grid['keraslr']))
    
    paramdraws.append(paramdraw)
    
#All the parameters are now sampled, we can create and store the models.
adastore = []
predsStore = []
devianceTestStore = []
devianceMeanStore = []
devianceFullStore = []

for i in range(0, nTests):
    print("test : ", i+1)
    params = paramdraws[i]
    estimator = AdaBoost(n_est=params['n_est'][0], loss = params['loss'][0], learning_rate=params['learning_rate'][0], kerasEpochs=params['kerasEpochs'][0],
                         kerasBatchSize=params['kerasBatchSize'][0], dropout = params['dropout'][0], nn1=params['nn1'][0], keraslr=params['keraslr'][0], 
                         input_dim=21)
    
    initWeights = estimator.initWeights(dataTrain)
    #fitting model
    estimator.fit(dataTrain, feed, initWeights)
    #predictions
    predSingle = estimator.predict(dataTest)
    #loss
    devSing = estimator.devFull(y1test, predSingle, d1test)
    meanDev = devSing/len(y1test)
    fullDev = meanDev * (len(y1) + len(y1test))
    
    devianceTestStore.append(devSing)
    devianceMeanStore.append(meanDev)
    devianceFullStore.append(fullDev)
    predsStore.append(predSingle)
    adastore.append(estimator)




#####################################
# PLOTS 
#####################################
#Learning curves : 
#take subsets of different size with chosen hyperparameters
#take mean of deviance because they are different size 
#####################################

#creating the subsets, then testing on subset train set and FULL test set.
#without duplicates !!

import matplotlib.pyplot as plt
import seaborn as sn

Xsubsets = []
Ysubsets = []
feedSubsets = []
dSubsets = []
trError = []
teError = []

feed = pd.DataFrame(feed)

#subsets fill 
nsubs = [100, 500, 1000, 2500, 5000, 7500, 10000, 30000, 51600]
for nsub in nsubs:
    tempTrain = dataTrain.sample(n = nsub, replace = False, random_state=24202, axis = 0)
    tempY = y1.sample(n = nsub, replace = False, random_state=24202, axis = 0)
    tempFeed = feed.sample(n = nsub, replace = False, random_state = 24202, axis = 0)
    tempD = d1.sample(n = nsub, replace = False, random_state = 24202, axis = 0)
    Xsubsets.append(tempTrain)
    Ysubsets.append(tempY)
    feedSubsets.append(tempFeed)
    dSubsets.append(tempD)
    
#Create model, calculate loss and apend 

#static params, other than nlength for instance
staticparams = {
            'n_est' : [5],
            'loss' : ['exponential'],
            'learning_rate' : [1],
            'kerasEpochs' : [10],
            'kerasBatchSize' : [64501],
            'dropout' : [0.2],
            'nn1' : [15],
            'keraslr' : [0.15],
        }



for i in range(0, len(nsubs)):
    params = staticparams
    estimator = AdaBoost(n_est=params['n_est'][0], loss = params['loss'][0], learning_rate=params['learning_rate'][0], kerasEpochs=params['kerasEpochs'][0],
                         kerasBatchSize=params['kerasBatchSize'][0], dropout = params['dropout'][0], nn1=params['nn1'][0], keraslr=params['keraslr'][0], 
                         input_dim=21)
    
    tempX = Xsubsets[i]
    tempY = Ysubsets[i]
    tempFeed = feedSubsets[i]
    tempD = dSubsets[i]
    
    tempInitWeights = estimator.initWeights(tempX)
    estimator.fit(tempX, tempFeed, tempInitWeights)
    
    tempPredTrain = estimator.predict(tempX)
    tempPredTest = estimator.predict(dataTest)
    
    #losses train
    tempLossTrain = estimator.devFull(tempY, tempPredTrain, tempD)
    meanLossTrain = tempLossTrain/len(tempY)
    
    #losses test
    tempLossTest = estimator.devFull(y1test, tempPredTest, d1test)
    meanLossTest = tempLossTest/len(y1test)
    
    #append
    teError.append(meanLossTest)
    trError.append(meanLossTrain)
    '''
    tempResults = tempGrid.cv_results_
    meanTestScore = tempResults['mean_test_score'][0]
    meanTrainScore = tempResults['mean_train_score'][0]
    teError.append(meanTestScore)
    trError.append(meanTrainScore)    
    '''
    
    
    
plt.plot(nsubs, teError, label = "Test Error")
plt.plot(nsubs, trError, label = "Train Error")
plt.xlabel('n_samples')
plt.ylabel('Deviance')
plt.title("Courbes d'apprentissage AdaBoost.R2")
plt.legend()
plt.savefig(root + '/lyx/images/learning/ADBnsamples.png')
plt.show()
plt.close()
    
#################################
#Plot for nn1
#################################

nn1s = [3,5,8,10,15,20,25,30,50,500]
teErrorNN = []
trErrorNN = []

#create static params
staticparamsList = []

for i in range(0, len(nn1s)):
    temp = {
            'n_est' : [5],
            'loss' : ['exponential'],
            'learning_rate' : [1],
            'kerasEpochs' : [100],
            'kerasBatchSize' : [64501],
            'dropout' : [0.2],
            'nn1' : [nn1s[i]],
            'keraslr' : [0.15],
        }
    staticparamsList.append(temp)
    


for i in range(0, len(nn1s)):
    params = staticparamsList[i]
    estimator = AdaBoost(n_est=params['n_est'][0], loss = params['loss'][0], learning_rate=params['learning_rate'][0], kerasEpochs=params['kerasEpochs'][0],
                         kerasBatchSize=params['kerasBatchSize'][0], dropout = params['dropout'][0], nn1=params['nn1'][0], keraslr=params['keraslr'][0], 
                         input_dim=21)
    
    initw = estimator.initWeights(dataTrain)
    estimator.fit(dataTrain, feed, initw)
    
    tempPredTrain = estimator.predict(dataTrain)
    tempPredTest = estimator.predict(dataTest)
    
    #losses train
    tempLossTrain = estimator.devFull(y1, tempPredTrain, d1)
    meanLossTrain = tempLossTrain/len(y1)
    
    #losses test
    tempLossTest = estimator.devFull(y1test, tempPredTest, d1test)
    meanLossTest = tempLossTest/len(y1test)
    
    #append
    teErrorNN.append(meanLossTest)
    trErrorNN.append(meanLossTrain)


plt.plot(nn1s, teErrorNN, label = "Test Error")
plt.plot(nn1s, trErrorNN, label = "Train Error")
plt.xlabel('n_neurons_1')
plt.ylabel('Deviance')
plt.title("Courbes d'apprentissage AdaBoost.R2")
plt.legend()
plt.savefig(root + '/lyx/images/learning/ADBnn1.png')
plt.show()
plt.close()

#################################
#Plot for LR adaboost
#################################

lrsad = [0.1,0.5,0.75,1,2]
teErrorLRad = []
trErrorLRad = []

#create static params
staticparamsListLR = []

for i in range(0, len(lrsad)):
    temp = {
            'n_est' : [5],
            'loss' : ['exponential'],
            'learning_rate' : [lrsad[i]],
            'kerasEpochs' : [10],
            'kerasBatchSize' : [64501],
            'dropout' : [0.2],
            'nn1' : [10],
            'keraslr' : [0.15],
        }
    staticparamsListLR.append(temp)


for i in range(0, len(lrsad)):
    params = staticparamsListLR[i]
    estimator = AdaBoost(n_est=params['n_est'][0], loss = params['loss'][0], learning_rate=params['learning_rate'][0], kerasEpochs=params['kerasEpochs'][0],
                         kerasBatchSize=params['kerasBatchSize'][0], dropout = params['dropout'][0], nn1=params['nn1'][0], keraslr=params['keraslr'][0], 
                         input_dim=21)
    
    initw = estimator.initWeights(dataTrain)
    estimator.fit(dataTrain, feed, initw)
    
    tempPredTrain = estimator.predict(dataTrain)
    tempPredTest = estimator.predict(dataTest)
    
    #losses train
    tempLossTrain = estimator.devFull(y1, tempPredTrain, d1)
    meanLossTrain = tempLossTrain/len(y1)
    
    #losses test
    tempLossTest = estimator.devFull(y1test, tempPredTest, d1test)
    meanLossTest = tempLossTest/len(y1test)
    
    #append
    teErrorLRad.append(meanLossTest)
    trErrorLRad.append(meanLossTrain)


plt.plot(lrsad, teErrorLRad, label = "Test Error")
plt.plot(lrsad, trErrorLRad, label = "Train Error")
plt.xlabel('Learning Rate ADB')
plt.ylabel('Deviance')
plt.title("Courbes d'apprentissage AdaBoost.R2")
plt.legend()
plt.savefig(root + '/lyx/images/learning/ADBLR.png')
plt.show()
plt.close()

#################################
#Plot for epochs
#################################

epochsList = [5,10,20,50,100,200,300,500,1000]
teErrorEpochs = []
trErrorEpochs = []

#create static params
staticparamsListEpochs = []

for i in range(0, len(epochsList)):
    temp = {
            'n_est' : [5],
            'loss' : ['exponential'],
            'learning_rate' : [1],
            'kerasEpochs' : [epochsList[i]],
            'kerasBatchSize' : [64501],
            'dropout' : [0.2],
            'nn1' : [10],
            'keraslr' : [0.15],
        }
    staticparamsListEpochs.append(temp)
    

for i in range(0, len(epochsList)):
    params = staticparamsListEpochs[i]
    estimator = AdaBoost(n_est=params['n_est'][0], loss = params['loss'][0], learning_rate=params['learning_rate'][0], kerasEpochs=params['kerasEpochs'][0],
                         kerasBatchSize=params['kerasBatchSize'][0], dropout = params['dropout'][0], nn1=params['nn1'][0], keraslr=params['keraslr'][0], 
                         input_dim=21)
    
    initw = estimator.initWeights(dataTrain)
    estimator.fit(dataTrain, feed, initw)
    
    tempPredTrain = estimator.predict(dataTrain)
    tempPredTest = estimator.predict(dataTest)
    
    #losses train
    tempLossTrain = estimator.devFull(y1, tempPredTrain, d1)
    meanLossTrain = tempLossTrain/len(y1)
    
    #losses test
    tempLossTest = estimator.devFull(y1test, tempPredTest, d1test)
    meanLossTest = tempLossTest/len(y1test)
    
    #append
    teErrorEpochs.append(meanLossTest)
    trErrorEpochs.append(meanLossTrain)


plt.plot(epochsList, teErrorEpochs, label = "Test Error")
plt.plot(epochsList, trErrorEpochs, label = "Train Error")
plt.xlabel('Epochs')
plt.ylabel('Deviance')
plt.title("Courbes d'apprentissage AdaBost.R2")
plt.legend()
plt.savefig(root + '/lyx/images/learning/ADBEpochs.png')
plt.show()
plt.close()











