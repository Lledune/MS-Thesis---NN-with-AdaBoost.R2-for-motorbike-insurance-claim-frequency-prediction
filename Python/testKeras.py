import tensorflow 
import keras 
import numpy as np 
import pandas as pd
from keras.layers import Dense
import keras.backend as KB
import math
data = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/mcc.csv")
#data2 = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocDataTrain.csv")
data2 = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocNoD.csv")
#y2 = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/NumberClaimsTrain.csv")
y2 = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/yNoD.csv")
#d = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/DurationTrain.csv")
d = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/durNoD.csv")
data['Gender'] = data['Gender'].map({'M' : 1, 'K' : 0})

data2['Duration'] = d

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

#def deviance metric testing
def DevianceBis(y_true, y_pred):
    y_pred = KB.maximum(y_pred, 0.0 + KB.epsilon()) #make sure ypred is positive or ln(-x) = NAN
    mask = KB.equal(y_true, 0)
    mask = KB.cast(mask, KB.floatx())
    return KB.sum(  (KB.sqrt(KB.square( 2 * KB.log(y_true + KB.epsilon()) - KB.log(y_pred)))))

model.compile(loss = Deviance_loss(), optimizer = 'RMSprop', metrics = [Deviance, "mean_squared_error"])
model.fit(factorsTrain, yTrain, epochs = 5)
model.fit(data2, y2, epochs = 5)

model.compile(loss = lossO, optimizer = 'RMSprop', metrics = [Deviance])
model.fit(factorsTrain, yTrain, epochs = 5)
model.fit(data2, y2, epochs = 5)


################### testing
model.compile(loss = DevianceBis, optimizer = 'RMSprop', metrics = [Deviance])
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

 import math 

    sgd = keras.optimizers.rmsprop(lr=0.01, clipnorm=1)

    def custom_loss(data, y_pred):

        y_true = data[:, 0]
        d = data[:, 1]
        # condition
        y_true = 100* y_true
        mask2 = keras.backend.greater(y_true, 0)
        mask2 = KB.cast(mask2, 'float32')

        # returns 0 when y_true =0, 1 otherwise
        #calculate loss using d...
        loss_value = KB.sqrt(KB.square(2 * d * y_pred + mask2  * (2 * d * y_true * KB.log(y_true + KB.epsilon()) + 2 * d * y_true * KB.log(y_pred + KB.epsilon()) - 2 * d * y_true)))
        return loss_value

    
    def baseline_model():
        # create model
        #building model
        model = keras.Sequential()
        model.add(Dense(5, input_dim = 26, activation = "relu"))
        #model.add(Dense(10, activation = "relu"))
        model.add(Dense(1, activation = "exponential"))
        model.compile(loss=custom_loss, optimizer='RMSProp')
        return model

model = baseline_model()
model.fit(data2, np.append(y2, d, axis = 1), epochs=2, shuffle=True, verbose=1)
model.fit(data2, y2, epochs=1, shuffle=True, verbose=1)
preds = model.predict(data2)
preds[4154]
