import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as KB
from math import log
from keras.layers import Dense, Dropout
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import _num_samples
import pickle
import matplotlib.pyplot as plt
import seaborn as sn

#########################################
#!!! change root to your main folder !!
#########################################
root = 'c:/users/kryst/desktop/poisson/poisson-neural-network-insurance-pricing'

###############################
# Importing datasets
###############################
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

#########################################
# AdaBoost.R2 CLASS modified
#########################################

class AdaBoostAlpha():
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
        """
        Creating adaboost object
        """
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
        """
        loss function used for keras optimisation
        """
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
        """
        Calculate deviance for a single observation
        """
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
        """
        Baseline model used for neural networks.
        The parameters are configurable
        This will use GPU so if you do not have cuda installed and working with keras, it might not work.
        """
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
        """
        Create the first vector of weight for adaboost training
        """
        n = len(data)
        weights = np.ones(n)
        weights = weights/sum(weights)
        return weights 
        
    
    #Normalizing the weights to sum = 1
    def normalizeWeights(self, weights):
        """
        Normalizes the weight so it is a distribution
        """
        sumW = sum(weights)
        newW = weights/sumW
        return newW
    
    #Resample data given weights, returns indices of said new data.
    def resampleIndices(self, data, weights):
        """
        Sample data based on weights
        """
        nRows = len(data)
        indices = np.arange(nRows)
        res = np.random.choice(indices, size = nRows, replace = True, p = weights)
        return res
    
    #Selecting rows with array of indices
    def dataFromIndices(self, data, indices):
        """
        Retrieves rows of given indices in a dataframe
        """
        newD = data.iloc[indices, :]
        return newD
    
    def printeq(self):
        print('==================================================')
    
    
    #The reason feed is passed instead of Y is because keras needs a tuple (y, d) to calculate the custom loss function deviance.
    #data is X
    def boost(self, iboost, data, feed, weights):
        """
        One iteration of boosting
        """
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
        
        self.averageLoss.append(estimator_error)    
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
        """
        Returns the model's prediction.
        """
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
        """
        Return model's prediction.
        """
        return self.getMedianPred(data, len(self.estimators))
    
    def stagedPredict(self, data):
        for i, _ in enumerate(self.estimators, 1):
            yield self.getMedianPred(data, limit = i)
            
    def save_model(self, path):
        """
        Saves the model to given path
        """
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
    def load_model(self, pathADB, nModels = 10):
        """
        Loads model from given path
        """
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
        for i in range(0, nModels):
            print(i)
            modelString = '/model' + str(i+1)
            modelFullString = modelsPath + modelString
            
            model = KerasRegressor(build_fn=self.baseline_model, epochs = 300, batch_size = 10, verbose = 2)
            model.model = keras.models.load_model(modelFullString, custom_objects={'deviance' : self.deviance})
            estLoaded.append(model)
        self.estimators = estLoaded
        
        print("Model loaded.")
        
def getParams(ada = AdaBoostAlpha()):
    """
    Returns a dictionary of parameters and evaluation metrics for a given AdaBoost object.
    """
    config_dic = {}
    config_dic['n_est'] = ada.n_est
    config_dic['loss'] = ada.loss
    config_dic['estimatorsErrors'] = ada.estimatorsErrors
    config_dic['estimatorsWeights'] = ada.estimatorsWeights
    config_dic['estimatorsSampleWeights'] = ada.estimatorsSampleWeights
    config_dic['learning_rate'] = ada.learning_rate
    config_dic['averageLoss'] = ada.averageLoss
    config_dic['kerasEpochs'] = ada.kerasEpochs
    config_dic['kerasBatchSize'] = ada.kerasBatchSize
    config_dic['dropout'] = ada.dropout
    config_dic['nn1'] = ada.nn1
    config_dic['keraslr'] = ada.keraslr
    config_dic['input_dim'] = ada.input_dim
    #adding the losses
    preds = ada.predict(dataTest)
    testLoss = ada.devFull(y1test, preds, d1test)
    meanTestLoss = testLoss/len(preds)
    fullTestLoss = meanTestLoss * 64501
    config_dic['testLoss'] = testLoss 
    config_dic['meanTestLoss'] = meanTestLoss
    config_dic['fullTestLoss'] = fullTestLoss
    config_dic['real_n_est'] = len(ada.estimators)
    return config_dic

#########################################
# AdaBoost.R2 CLASS (original)
#########################################

class AdaBoostBeta():
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
        """
        Creating adaboost object
        """
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
        """
        loss function used for keras optimisation
        """
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
        """
        Calculate deviance for a single observation
        """
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
        """
        Baseline model used for neural networks.
        The parameters are configurable
        This will use GPU so if you do not have cuda installed and working with keras, it might not work.
        """
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
        """
        Create the first vector of weight for adaboost training
        """
        n = len(data)
        weights = np.ones(n)
        weights = weights/sum(weights)
        return weights 
        
    
    #Normalizing the weights to sum = 1
    def normalizeWeights(self, weights):
        """
        Normalizes the weight so it is a distribution
        """
        sumW = sum(weights)
        newW = weights/sumW
        return newW
    
    #Resample data given weights, returns indices of said new data.
    def resampleIndices(self, data, weights):
        """
        Sample data based on weights
        """
        nRows = len(data)
        indices = np.arange(nRows)
        res = np.random.choice(indices, size = nRows, replace = True, p = weights)
        return res
    
    #Selecting rows with array of indices
    def dataFromIndices(self, data, indices):
        """
        Retrieves rows of given indices in a dataframe
        """
        newD = data.iloc[indices, :]
        return newD
    
    def printeq(self):
        print('==================================================')
    
    
    #The reason feed is passed instead of Y is because keras needs a tuple (y, d) to calculate the custom loss function deviance.
    #data is X
    def boost(self, iboost, data, feed, weights):
        """
        One iteration of boosting
        """
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
        
        self.averageLoss.append(estimator_error)    
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
        estimator_weight = beta
        
        if not iboost == self.n_est - 1:
            weights[sample_mask] *= np.power(
                beta, (1.-masked_error_vector)
            )
                    
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
        """
        Returns the model's prediction.
        """
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
        """
        Return model's prediction.
        """
        return self.getMedianPred(data, len(self.estimators))
    
    def stagedPredict(self, data):
        for i, _ in enumerate(self.estimators, 1):
            yield self.getMedianPred(data, limit = i)
            
    def save_model(self, path):
        """
        Saves the model to given path
        """
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
    def load_model(self, pathADB, nModels = 10):
        """
        Loads model from given path
        """
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
        for i in range(0, nModels):
            print(i)
            modelString = '/model' + str(i+1)
            modelFullString = modelsPath + modelString
            
            model = KerasRegressor(build_fn=self.baseline_model, epochs = 300, batch_size = 10, verbose = 2)
            model.model = keras.models.load_model(modelFullString, custom_objects={'deviance' : self.deviance})
            estLoaded.append(model)
        self.estimators = estLoaded
        
        print("Model loaded.")
        
def getParams(ada = AdaBoostBeta()):
    """
    Returns a dictionary of parameters and evaluation metrics for a given AdaBoost object.
    """
    config_dic = {}
    config_dic['n_est'] = ada.n_est
    config_dic['loss'] = ada.loss
    config_dic['estimatorsErrors'] = ada.estimatorsErrors
    config_dic['estimatorsWeights'] = ada.estimatorsWeights
    config_dic['estimatorsSampleWeights'] = ada.estimatorsSampleWeights
    config_dic['learning_rate'] = ada.learning_rate
    config_dic['averageLoss'] = ada.averageLoss
    config_dic['kerasEpochs'] = ada.kerasEpochs
    config_dic['kerasBatchSize'] = ada.kerasBatchSize
    config_dic['dropout'] = ada.dropout
    config_dic['nn1'] = ada.nn1
    config_dic['keraslr'] = ada.keraslr
    config_dic['input_dim'] = ada.input_dim
    #adding the losses
    preds = ada.predict(dataTest)
    testLoss = ada.devFull(y1test, preds, d1test)
    meanTestLoss = testLoss/len(preds)
    fullTestLoss = meanTestLoss * 64501
    config_dic['testLoss'] = testLoss 
    config_dic['meanTestLoss'] = meanTestLoss
    config_dic['fullTestLoss'] = fullTestLoss
    config_dic['real_n_est'] = len(ada.estimators)
    return config_dic


#keep learning to 1 for beta version or it is not correct !!!
adbb = AdaBoostBeta(n_est=20, loss = 'exponential', learning_rate = 1, kerasEpochs = 150, kerasBatchSize=51600, dropout = 0.1, nn1 = 5,
                    keraslr = 0.5)
initWeights = adbb.initWeights(dataTrain)
adbb.fit(dataTrain, feed, initWeights)

adba1 = AdaBoostAlpha(n_est=20, loss = 'exponential', learning_rate = 1, kerasEpochs = 150, kerasBatchSize=51600, dropout = 0.1, nn1 = 5,
                    keraslr = 0.5)
adba1.fit(dataTrain, feed, initWeights)

adba2 = AdaBoostAlpha(n_est=20, loss = 'exponential', learning_rate = 0.1, kerasEpochs = 150, kerasBatchSize=51600, dropout = 0.1, nn1 = 5,
                    keraslr = 0.5)
adba2.fit(dataTrain, feed, initWeights)

adba3 = AdaBoostAlpha(n_est=20, loss = 'exponential', learning_rate = 0.5, kerasEpochs = 150, kerasBatchSize=51600, dropout = 0.1, nn1 = 5,
                    keraslr = 0.5)
adba3.fit(dataTrain, feed, initWeights)


#plotting evolution of estimator weights
weightsAlpha1 = adba1.estimatorsWeights
weightsAlpha2 = adba2.estimatorsWeights
weightsAlpha3 = adba3.estimatorsWeights
weightsBeta = adbb.estimatorsWeights

rangeAlpha1 = range(0, len(weightsAlpha1))
rangeAlpha2 = range(0, len(weightsAlpha2))
rangeAlpha3 = range(0, len(weightsAlpha3))
rangeBeta = range(0, len(weightsBeta))

plt.plot(rangeAlpha1, weightsAlpha1, label = 'LR = 1', alpha = 0.5)
plt.plot(rangeAlpha3, weightsAlpha3, label = 'LR = 0.5', alpha = 0.5)
plt.plot(rangeAlpha2, weightsAlpha2, label = 'LR = 0.1', alpha = 0.5)
plt.plot(rangeBeta, weightsBeta, label = 'Version originale', linestyle = 'dashed')
plt.title("Comparaison des différentes versions d'AdaBoost.R2")
plt.ylabel("Importance de l'apprenant faible")
plt.xlabel("Itérations")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#Plotting evolution of sample weight (n = 0)
weightsSampAlpha1 = adba1.estimatorsSampleWeights
weightsSampAlpha2 = adba2.estimatorsSampleWeights
weightsSampAlpha3 = adba3.estimatorsSampleWeights
weightsSampBeta = adbb.estimatorsSampleWeights

alpha1data = []
alpha2data = []
alpha3data = []
betadata = []

for vector in weightsSampAlpha1:
    alpha1data.append(vector[0])
    
for vector in weightsSampAlpha2:
    alpha2data.append(vector[0])

for vector in weightsSampAlpha3:
    alpha3data.append(vector[0])
    
for vector in weightsSampBeta:
    betadata.append(vector[0])
    
rangeAlpha1 = range(0, len(alpha1data))
rangeAlpha2 = range(0, len(alpha2data))
rangeAlpha3 = range(0, len(alpha3data))
rangeBeta = range(0, len(betadata))

    
    

plt.plot(rangeAlpha1, alpha1data, label = "LR = 1", alpha = 0.5)
plt.plot(rangeAlpha3, alpha3data, label = "LR = 0.5", alpha = 0.5)
plt.plot(rangeAlpha2, alpha2data, label = "LR = 0.1", alpha = 0.5)
plt.plot(rangeBeta, betadata, label = "Version originale", linestyle = 'dashed')
plt.title("Évolution d'un poids")
plt.ylabel("Valeur du poids")
plt.xlabel("Itérations")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


#plotting evolution of estimator weights
weightsAlpha1 = adba1.averageLoss
weightsAlpha2 = adba2.averageLoss
weightsAlpha3 = adba3.averageLoss
weightsBeta = adbb.averageLoss

rangeAlpha1 = range(0, len(weightsAlpha1))
rangeAlpha2 = range(0, len(weightsAlpha2))
rangeAlpha3 = range(0, len(weightsAlpha3))
rangeBeta = range(0, len(weightsBeta))

plt.plot(rangeAlpha1, weightsAlpha1, label = 'LR = 1', alpha = 0.5)
plt.plot(rangeAlpha3, weightsAlpha3, label = 'LR = 0.5', alpha = 0.5)
plt.plot(rangeAlpha2, weightsAlpha2, label = 'LR = 0.1', alpha = 0.5)
plt.plot(rangeBeta, weightsBeta, label = 'Version originale', linestyle = 'dashed')
plt.title("Comparaison des différentes versions d'AdaBoost.R2")
plt.ylabel("Erreur moyenne")
plt.xlabel("Itérations")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


