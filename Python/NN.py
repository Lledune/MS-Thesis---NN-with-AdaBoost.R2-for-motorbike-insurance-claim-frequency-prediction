#############################################
# This file contains the shallow NN         #
# implementation, the parameter search      #
# and the learning curves plots.            #
#############################################


#NN file 
import tensorflow as tf
import keras 
import numpy as np 
import pandas as pd
import keras.backend as KB
from math import log
from sklearn.model_selection import KFold
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


#########################################
#!!! change root to your main folder !!
#########################################
root = 'c:/users/kryst/desktop/poisson/poisson-neural-network-insurance-pricing'


#Loss function     
def deviance(data, y_pred):
    """
    Used as the loss function for NN weights optimization

    Parameters
    ----------
    data : concatenate of y and duration
        
    y_pred : the prediction (used by keras)

    Returns
    -------
    loss : Loss value for keras
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

def devSingle(y, yhat, d):
    """
    Used to get the deviance of a single prediction

    Parameters
    ----------
    y : real value
    yhat : predicted value
    d : duration

    Returns
    -------
    float
        deviance result

    """
    if y != 0:
        return 2 * d * (y * log(y) - y * log(yhat) - y + yhat)
    else:
        return 2 * d * yhat
    return print("error")

def devFull(y, yhat, d):
    """
    Used to get the deviance of a vector of prediction

    Parameters
    ----------
    y : real value
    yhat : predicted value
    d : duration

    Returns
    -------
    float
        deviance result (sum)

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
        dev = devSingle(row["y"], row["yhat"], row["d"])
        sumtot = sumtot + dev
    return sumtot

#used to check that keras is well using GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def baseline_model2(dropout = 0.2, kernel_initializer = 'glorot_uniform', nn1 = 15, lr = 0.001, act1 = "relu"):
    """
    Used as the baseline model for keras (shallow network)
    
    Parameters
    ----------
    dropout : float, optional
        Dropout rate. The default is 0.2.
    kernel_initializer : string, optional
        The default is 'glorot_uniform'.
    nn1 : int, optional
        Number of neurons. The default is 15.
    lr : float, optional
        Learning rate. The default is 0.001. [0,1]
    act1 : string, optional
        Activation function of hidden layer. The default is "relu".

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    with tf.device('/gpu:0'): #use the GPU
        # create model
        #building model
        model = keras.Sequential()
        model.add(Dense(nn1, input_dim = 21, activation = act1, kernel_initializer=kernel_initializer))
        model.add(Dropout(dropout))
        #model.add(Dense(2, activation = "exponential"))
        #model.add(Dense(10, activation = "relu"))
        model.add(Dense(1, activation = "exponential", kernel_initializer=kernel_initializer))
        optimizer = keras.optimizers.adagrad(lr=lr)
        model.compile(loss=deviance, optimizer=optimizer, metrics = [deviance, "mean_squared_error"])
        return model



#########################################
# Importing the data
#########################################


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




##############################################
# Parameters search : crossvalidation ########
##############################################


clf = KerasRegressor(build_fn=baseline_model2)

#hyperparameters domain
param_grid = {
    'clf__epochs':[250,400,100],
    'clf__dropout':[0.1,0.2,0.3],
    'clf__kernel_initializer':['uniform'],
    'clf__batch_size':[51600, 10000, 1000,500],
    'clf__nn1':[5,10,15,25],
    'clf__lr':[0.1,0.3,0.05,0.01],
    'clf__act1':['softmax']
}

pipeline = Pipeline([
    ('clf',clf)
])

#cross validation, only 4 because the computation time is huge. 
cv = KFold(n_splits=4, shuffle=False)
grid = RandomizedSearchCV(pipeline, cv = cv, param_distributions=param_grid, verbose=3, n_iter = 40) #plus de folds pourraient augmenter la variance
grid.fit(dataTrain, feed)

results = pd.DataFrame(grid.cv_results_)
results.to_csv(root + '/NNshallowCV.csv')
best = grid.best_estimator_

ypredtest = best.predict(dataTest)
ypredtrain = best.predict(dataTrain)
devTest = devFull(y1test, ypredtest, d1test)
devTrain = devFull(y1, ypredtrain, d1)
meanDevTest = devTest/len(y1test)
meanDevTrain = devTrain/len(y1)

#the normalized deviances
totDevTrain = meanDevTrain * (len(y1) + len(y1test))
totDevTest = meanDevTest * (len(y1) + len(y1test))


#based on the code up to this comment, the chosen parameters are 
#nn1 : 25
#lr : 0.1
#kernel : uniform
#epochs : 250
#batch_size :500
#act1 : softmax
#dropout : 0.1


######################################
#SAVE MODEL 
######################################

#getting model from pipeline
model_to_save = best.named_steps['clf'].model

#saving model
model_to_save.save(root + '/Python/Models/NNmodel')



######################################
#LOAD MODEL
#NEED TO ADD THE CUSTOM LOSS AS OBJECT IN LOAD !! Like this :
######################################
reconstructed_model = keras.models.load_model(root + '/Python/Models/NNmodel', custom_objects={'deviance' : deviance})

preds = reconstructed_model.predict(dataTest)
devTest = devFull(y1test, preds, d1test)
devMean = devTest/len(y1test)
devFull = devMean * 64501



#####################################
# PLOTS 
#####################################
#Learning curves : 
#take subsets of different size with chosen hyperparameters
#take mean of deviance because they are different size 
#####################################

#####################################
#Plot for n_samples 
#####################################


#creating the subsets, then testing on subset train set and FULL test set.
#without duplicates !!

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
    'clf__epochs':[250],
    'clf__dropout':[0.1],
    'clf__kernel_initializer':['uniform'],
    'clf__batch_size':[500],
    'clf__nn1':[15],
    'clf__lr':[0.1],
    'clf__act1':['softmax']
}

cv2 = KFold(n_splits=2, shuffle=False)



for i in range(0, len(nsubs)):
    tempModel = KerasRegressor(build_fn=baseline_model2, verbose = 1)
    tempPipeline = Pipeline([('clf',tempModel)])
    tempGrid = RandomizedSearchCV(tempPipeline, cv = cv2, param_distributions=staticparams, verbose = 0, n_iter=1, return_train_score=False)
    
    tempX = Xsubsets[i]
    tempY = Ysubsets[i]
    tempFeed = feedSubsets[i]
    tempD = dSubsets[i]
    
    tempGrid.fit(tempX, tempFeed)
    tempBest = tempGrid.best_estimator_
    
    tempPredTrain = tempBest.predict(tempX)
    tempPredTest = tempBest.predict(dataTest)
    
    #losses train
    tempLossTrain = devFull(tempY, tempPredTrain, tempD)
    meanLossTrain = tempLossTrain/len(tempY)
    
    #losses test
    tempLossTest = devFull(y1test, tempPredTest, d1test)
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
plt.xlabel('N. échantillons')
plt.ylabel('Deviance')
plt.title("Courbes d'apprentissage NN")
plt.legend()
plt.savefig(root + '/lyx/images/learning/NNnsamples.png')
plt.show()
plt.close()
    
#################################
#Plot for nn1
#################################

nn1s = [3,5,8,10,15,20,25,30,50,500,1000,2500]
teErrorNN = []
trErrorNN = []

#create static params
staticparamsList = []

for i in range(0, len(nn1s)):
    temp = param_grid2 = {
    'clf__epochs':[250],
    'clf__dropout':[0.1],
    'clf__kernel_initializer':['uniform'],
    'clf__batch_size':[500],
    'clf__nn1':[nn1s[i]],
    'clf__lr':[0.1],
    'clf__act1':['softmax']
}
    staticparamsList.append(temp)
    
cv3 = KFold(n_splits=2, shuffle=False)


for i in range(0, len(nn1s)):
    tempModel = KerasRegressor(build_fn=baseline_model2, verbose = 1)
    tempPipeline = Pipeline([('clf',tempModel)])
    tempGrid = RandomizedSearchCV(tempPipeline, cv = cv3, param_distributions=staticparamsList[i], verbose = 0, n_iter=1, return_train_score=False)
    
    tempGrid.fit(dataTrain, feed)
    tempBest = tempGrid.best_estimator_
    
    tempPredTrain = tempBest.predict(dataTrain)
    tempPredTest = tempBest.predict(dataTest)
    
    #losses train
    tempLossTrain = devFull(y1, tempPredTrain, d1)
    meanLossTrain = tempLossTrain/len(y1)
    
    #losses test
    tempLossTest = devFull(y1test, tempPredTest, d1test)
    meanLossTest = tempLossTest/len(y1test)
    
    #append
    teErrorNN.append(meanLossTest)
    trErrorNN.append(meanLossTrain)


plt.plot(nn1s, teErrorNN, label = "Test Error")
plt.plot(nn1s, trErrorNN, label = "Train Error")
plt.xlabel('N. neurones')
plt.ylabel('Deviance')
plt.title("Courbes d'apprentissage NN")
plt.legend()
plt.savefig(root + '/lyx/images/learning/NNnn1.png')
plt.show()
plt.close()


#################################
#Plot for LR
#################################

lrs = [0.001, 0.01, 0.1, 0.15, 0.20]
teErrorLR = []
trErrorLR = []

#create static params
staticparamsListLR = []

for i in range(0, len(lrs)):
    temp = param_grid2 = {
    'clf__epochs':[250],
    'clf__dropout':[0.1],
    'clf__kernel_initializer':['uniform'],
    'clf__batch_size':[500],
    'clf__nn1':[15],
    'clf__lr':[lrs[i]],
    'clf__act1':['softmax']
}
    staticparamsListLR.append(temp)
    
cv3 = KFold(n_splits=2, shuffle=False)


for i in range(0, len(lrs)):
    tempModel = KerasRegressor(build_fn=baseline_model2, verbose = 1)
    tempPipeline = Pipeline([('clf',tempModel)])
    tempGrid = RandomizedSearchCV(tempPipeline, cv = cv3, param_distributions=staticparamsListLR[i], verbose = 0, n_iter=1, return_train_score=False)
    
    tempGrid.fit(dataTrain, feed)
    tempBest = tempGrid.best_estimator_
    
    tempPredTrain = tempBest.predict(dataTrain)
    tempPredTest = tempBest.predict(dataTest)
    
    #losses train
    tempLossTrain = devFull(y1, tempPredTrain, d1)
    meanLossTrain = tempLossTrain/len(y1)
    
    #losses test
    tempLossTest = devFull(y1test, tempPredTest, d1test)
    meanLossTest = tempLossTest/len(y1test)
    
    #append
    teErrorLR.append(meanLossTest)
    trErrorLR.append(meanLossTrain)


plt.plot(lrs, teErrorLR, label = "Test Error")
plt.plot(lrs, trErrorLR, label = "Train Error")
plt.xlabel("Taux d'apprentissage")
plt.ylabel('Deviance')
plt.title("Courbes d'apprentissage NN")
plt.legend()
plt.savefig(root + '/lyx/images/learning/NNLR.png')
plt.show()
plt.close()

#################################
#Plot for epochs
#################################

epochsList = [5,10,20,50,100,200,300,500]
teErrorEpochs = []
trErrorEpochs = []

#create static params
staticparamsListEpochs = []

for i in range(0, len(epochsList)):
    temp = {
    'clf__epochs':[epochsList[i]],
    'clf__dropout':[0.1],
    'clf__kernel_initializer':['uniform'],
    'clf__batch_size':[500],
    'clf__nn1':[15],
    'clf__lr':[0.1],
    'clf__act1':['softmax']
}
    staticparamsListEpochs.append(temp)
    
cv3 = KFold(n_splits=2, shuffle=False)


for i in range(0, len(epochsList)):
    tempModel = KerasRegressor(build_fn=baseline_model2, verbose = 1)
    tempPipeline = Pipeline([('clf',tempModel)])
    tempGrid = RandomizedSearchCV(tempPipeline, cv = cv3, param_distributions=staticparamsListEpochs[i], verbose = 0, n_iter=1, return_train_score=False)
    
    tempGrid.fit(dataTrain, feed)
    tempBest = tempGrid.best_estimator_
    
    tempPredTrain = tempBest.predict(dataTrain)
    tempPredTest = tempBest.predict(dataTest)
    
    #losses train
    tempLossTrain = devFull(y1, tempPredTrain, d1)
    meanLossTrain = tempLossTrain/len(y1)
    
    #losses test
    tempLossTest = devFull(y1test, tempPredTest, d1test)
    meanLossTest = tempLossTest/len(y1test)
    
    #append
    teErrorEpochs.append(meanLossTest)
    trErrorEpochs.append(meanLossTrain)


plt.plot(epochsList, teErrorEpochs, label = "Test Error")
plt.plot(epochsList, trErrorEpochs, label = "Train Error")
plt.xlabel('Époques')
plt.ylabel('Deviance')
plt.title("Courbes d'apprentissage NN")
plt.legend()
plt.savefig(root + '/lyx/images/learning/NNEpochs.png')
plt.show()
plt.close()


###################################
#Last model refining test
###################################

clf = KerasRegressor(build_fn=baseline_model2)

#hyperparameters domain
param_grid = {
    'clf__epochs':[125],
    'clf__dropout':[0.1],
    'clf__kernel_initializer':['uniform'],
    'clf__batch_size':[500],
    'clf__nn1':[50],
    'clf__lr':[0.1],
    'clf__act1':['softmax']
}

pipeline = Pipeline([
    ('clf',clf)
])

#cross validation, only 4 because the computation time is huge. 
cv = KFold(n_splits=4, shuffle=False)
grid = RandomizedSearchCV(pipeline, cv = cv, param_distributions=param_grid, verbose=3, n_iter = 1) 
grid.fit(dataTrain, feed)

resultsRedinedNN = pd.DataFrame(grid.cv_results_)
best = grid.best_estimator_

ypredtest = best.predict(dataTest)
ypredtrain = best.predict(dataTrain)
devTest = devFull(y1test, ypredtest, d1test)
devTrain = devFull(y1, ypredtrain, d1)
meanDevTest = devTest/len(y1test)
meanDevTrain = devTrain/len(y1)

#the normalized deviances
totDevTrain = meanDevTrain * (len(y1) + len(y1test))
totDevTest = meanDevTest * (len(y1) + len(y1test))



############################
# Test dev of best models
############################

#hyperparameters domain
param_grid1 = {
    'clf__epochs':[250],
    'clf__dropout':[0.1],
    'clf__kernel_initializer':['uniform'],
    'clf__batch_size':[500],
    'clf__nn1':[25],
    'clf__lr':[0.1],
    'clf__act1':['softmax']
}

#hyperparameters domain
param_grid2 = {
    'clf__epochs':[250],
    'clf__dropout':[0.1],
    'clf__kernel_initializer':['uniform'],
    'clf__batch_size':[500],
    'clf__nn1':[15],
    'clf__lr':[0.1],
    'clf__act1':['softmax']
}

#hyperparameters domain
param_grid3 = {
    'clf__epochs':[400],
    'clf__dropout':[0.1],
    'clf__kernel_initializer':['uniform'],
    'clf__batch_size':[500],
    'clf__nn1':[10],
    'clf__lr':[0.1],
    'clf__act1':['softmax']
}

clf = KerasRegressor(build_fn=baseline_model2)
pipeline = Pipeline([
    ('clf',clf)
])
cv = KFold(n_splits=2, shuffle=False)

grid1 = RandomizedSearchCV(pipeline, cv = cv, param_distributions=param_grid1, verbose=3, n_iter = 1) 
grid1.fit(dataTrain, feed)
grid2 = RandomizedSearchCV(pipeline, cv = cv, param_distributions=param_grid2, verbose=3, n_iter = 1) 
grid2.fit(dataTrain, feed)
grid3 = RandomizedSearchCV(pipeline, cv = cv, param_distributions=param_grid3, verbose=3, n_iter = 1) 
grid3.fit(dataTrain, feed)


best1 = grid1.best_estimator_
best2 = grid2.best_estimator_
best3 = grid3.best_estimator_

ypredtest1 = best1.predict(dataTest)
ypredtest2 = best2.predict(dataTest)
ypredtest3 = best3.predict(dataTest)

devTest1 = devFull(y1test, ypredtest1, d1test)
devTest2 = devFull(y1test, ypredtest2, d1test)
devTest3 = devFull(y1test, ypredtest3, d1test)

normDevTest1 = (devTest1/len(y1test))*(len(y1test) + len(y1))
normDevTest2 = (devTest2/len(y1test))*(len(y1test) + len(y1))
normDevTest3 = (devTest3/len(y1test))*(len(y1test) + len(y1))
meanDevTest1 = (devTest1/len(y1test))
meanDevTest2 = (devTest2/len(y1test))
meanDevTest3 = (devTest3/len(y1test))








