import tensorflow 
import keras 
import numpy as np 
import pandas as pd
from keras.layers import Dense
import keras.backend as KB
import math
data = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/mcc.csv")

data['Gender'] = data['Gender'].map({'M' : 1, 'K' : 0})


mdk = np.random.rand(len(data)) < 0.8



factors = data.iloc[:, 0:6]
y = data.iloc[:, 7]

factorsTrain = factors[mdk]
factorsTest = factors[~mdk]

yTrain = y[mdk]
yTest = y[~mdk]

#building model
model = keras.Sequential()
model.add(Dense(5, input_dim = 6, activation = "relu"))
model.add(Dense(10, activation = "relu"))
model.add(Dense(1, activation = "exponential"))

#DEF CUSTOM LOSS
def custom_loss():
    def loss(y_true, y_pred):
        y_true = KB.max(y_true, 0)
        return (KB.sqrt(KB.square( 2 * KB.log(y_true + KB.epsilon()) - KB.log(y_pred))))
    return loss

def lossO(y_true, y_pred):
    y_true = KB.max(y_true, 0)
    return (KB.sqrt(KB.square( 2 * KB.log(y_true + KB.epsilon()) - KB.log(y_pred))))

#def deviance metric
def custom_metrics(y_true, y_pred):
    return (KB.sqrt(KB.square( 2 * KB.log(y_true + KB.epsilon()) - KB.log(y_pred))))


model.compile(loss = custom_loss(), optimizer = 'RMSprop', metrics = [custom_metrics, "mean_squared_error"])
model.fit(factorsTrain, yTrain, epochs = 5)

model.compile(loss = lossO, optimizer = 'RMSprop', metrics = [custom_metrics])
model.fit(factorsTrain, yTrain, epochs = 5)

model.compile(loss = "poisson", optimizer = 'RMSprop', metrics = [custom_metrics])
model.fit(factorsTrain, yTrain, epochs = 20)

model.compile(loss = "mean_squared_error", optimizer = 'RMSprop', metrics = [custom_metrics, "mean_squared_error"])
model.fit(factorsTrain, yTrain, epochs = 20)

#Test results 
model.evaluate(factorsTrain, yTrain)
model.evaluate(factorsTest, yTest)

