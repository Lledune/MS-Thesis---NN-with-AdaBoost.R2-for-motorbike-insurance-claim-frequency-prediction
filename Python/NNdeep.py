#NN file 
import tensorflow as tf
import keras 
import numpy as np 
import pandas as pd
import keras.backend as KB
from math import log, exp
import os
from sklearn.model_selection import KFold, StratifiedKFold
from keras.layers import Dense, Dropout


#importing datasets
data = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/preprocFull.csv")
datacats = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/catOnly.csv")

#shuffling because the dataset is ordered by age and the young clients have more accidents which leads to unbalanced k-fold.
data = data.sample(frac = 1, random_state = 24202).reset_index(drop=True)
datacats = datacats.sample(frac = 1, random_state = 24202).reset_index(drop=True)

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

#used to check that keras is well using GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


####################
#Trying to insert keras into sklearn, and search hyperparameters
#This is the first iteration of it, we check the big domains of hyperparameter and will refine them afterwards depending on the results.
####################

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

def baseline_model2(dropout = 0.2, kernel_initializer = 'glorot_uniform', nn1 = 15, nn2 = 10, lr = 0.001, act1 = "relu"):
    with tf.device('/gpu:0'):
        # create model
        #building model
        model = keras.Sequential()
        model.add(Dense(nn1, input_dim = 21, activation = act1, kernel_initializer=kernel_initializer))
        model.add(Dropout(dropout))
        #model.add(Dense(2, activation = "exponential"))
        model.add(Dense(nn2, activation = act1))
        model.add(Dense(1, activation = "exponential", kernel_initializer=kernel_initializer))
        optimizer = keras.optimizers.adagrad(lr=lr)
        model.compile(loss=deviance, optimizer=optimizer, metrics = [deviance, "mean_squared_error"])
        return model


clf = KerasRegressor(build_fn=baseline_model2)

param_grid = {
    'clf__epochs':[300,600,100],
    'clf__dropout':[0.1,0.5],
    'clf__kernel_initializer':['uniform'],
    'clf__batch_size':[50000, 10000, 1000],
    'clf__nn1':[10,15,20],
    'clf__nn2':[15,20,25],
    'clf__lr':[0.1,0.2,0.3],
    'clf__act1':['exponential', 'softmax']
}

pipeline = Pipeline([
    ('clf',clf)
])

cv = KFold(n_splits=5, shuffle=False)

grid = RandomizedSearchCV(pipeline, cv = cv, param_distributions=param_grid, verbose=2, n_iter = 40) #plus de folds pourraient augmenter la variance
grid.fit(data, feed)

results2 = pd.DataFrame(grid.cv_results_)
results2.to_csv('NNdeepCV.csv')
best2 = grid.best_estimator_

ypred2 = best2.predict(data)
devTest = devFull(y1, ypred2, d1)

#TODO : TESTER AVEC DONATIEN DE METTRE EXP LINK FUNCTION et voir si les résultats sont meilleurs. 









