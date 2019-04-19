import tensorflow 
import keras 
import numpy as np 
import pandas as pd
from keras.layers import Dense
import keras.backend as KB
import math
data = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/mcc.csv")
data2 = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocData.csv")
y2 = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/NumberClaims.csv")
data['Gender'] = data['Gender'].map({'M' : 1, 'K' : 0})

data2

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
#model.add(Dense(10, activation = "relu"))
model.add(Dense(1, activation = "exponential"))

#building model 2 
model = keras.Sequential()
model.add(Dense(25, input_dim = 25, activation = "relu"))
#model.add(Dense(10, activation = "relu"))
model.add(Dense(1, activation = "exponential"))

#DEF CUSTOM LOSS
def Deviance_loss():
    def loss(y_true, y_pred):
        return (KB.sqrt(KB.square( 2 * KB.log(y_true + KB.epsilon()) - KB.log(y_pred + KB.epsilon()))))
    return loss

def lossO(y_true, y_pred):
    y_true = KB.max(y_true, 0)
    return (KB.sqrt(KB.square( 2 * KB.log(y_true + KB.epsilon()) - KB.log(y_pred))))

#def deviance metric
def Deviance(y_true, y_pred):
    y_pred = KB.maximum(y_pred, 0.0 + KB.epsilon()) #make sure ypred is positive or ln(-x) = NAN
    return (KB.sqrt(KB.square( 2 * KB.log(y_true + KB.epsilon()) - KB.log(y_pred))))

#def deviance metric
def DevianceBis(y_true, y_pred):
    y_pred = KB.maximum(y_pred, 0.0 + KB.epsilon()) #make sure ypred is positive or ln(-x) = NAN
    return (KB.sqrt(KB.square( 2 * KB.log(y_true + KB.epsilon()) - KB.log(y_pred))))

model.compile(loss = Deviance_loss(), optimizer = 'RMSprop', metrics = [Deviance, "mean_squared_error"])
model.fit(factorsTrain, yTrain, epochs = 5)
model.fit(data2, y2, epochs = 5)

model.compile(loss = lossO, optimizer = 'RMSprop', metrics = [Deviance])
model.fit(factorsTrain, yTrain, epochs = 5)
model.fit(data2, y2, epochs = 5)

model.compile(loss = "poisson", optimizer = 'RMSprop', metrics = [Deviance])
model.fit(factorsTrain, yTrain, epochs = 5)
model.fit(data2, y2, epochs = 5)

model.compile(loss = "mean_squared_error", optimizer = 'RMSprop', metrics = [Deviance, "mean_squared_error"])
model.fit(factorsTrain, yTrain, epochs = 5)
model.fit(data2, y2, epochs = 5)

#Test results 
model.evaluate(factorsTrain, yTrain)
model.evaluate(factorsTest, yTest)
#plots 
import matplotlib.pyplot as plt
plt.plot(history.history['Deviance'])
plt.title('Model Deviance')
plt.ylabel('Deviance')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
