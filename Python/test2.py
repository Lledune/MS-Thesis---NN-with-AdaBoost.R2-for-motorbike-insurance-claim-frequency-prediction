import tensorflow as tf
import keras 
import numpy as np 
import pandas as pd
from keras.layers import Dense
import keras.backend as KB
from math import log

data = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocFull.csv")

factors = data.drop('NumberClaims', axis = 1)
factors = factors.drop('ClaimFrequency', axis = 1)
nbClaims = data['NumberClaims']
nbClaims = nbClaims.to_frame()
y = data['ClaimFrequency']
d = data['Duration']
y = pd.DataFrame(y)
d = pd.DataFrame(d)
feed = np.append(y, d, axis = 1) #Used to pass d with y in loss function 
y = y.values
d = d.values

def custom_loss3(data, y_pred):
    y_true = data[:, 0]
    d = data[:, 1]
    
    lnYTrue = KB.switch(KB.equal(y_true, 0), KB.zeros_like(y_true), KB.log(y_true))
    lnYPred = KB.switch(KB.equal(y_pred, 0), KB.zeros_like(y_pred), KB.log(y_pred))
    loss_value = 2 * d * (y_true * lnYTrue - y_true * lnYPred[:, 0] - y_true + y_pred[:, 0])
    return loss_value
    
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


def baseline_model(loss):
    # create model
    #building model
    model = keras.Sequential()
    model.add(Dense(25, input_dim = 26, activation = "exponential"))

    #model.add(Dense(10, activation = "relu"))
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(loss=loss, optimizer='Adam')
    return model

model1 = baseline_model("mean_squared_error")
model2 = baseline_model(deviance)

model2.fit(factors, feed, epochs=4, shuffle=True, verbose=1)
model1.fit(factors, y, epochs=4, shuffle=True, verbose=1)

#model 1 = poisson, model 2 = custom
#In spyder IDE can check comp to see if two models are similar
preds1 = model1.predict(factors)
preds2 = model2.predict(factors)

compare1 = np.append(y, preds1, axis = 1)
compare1 = pd.DataFrame(compare1)

compare2 = np.append(y, preds2, axis = 1)
compare2 = pd.DataFrame(compare2)

comp = np.append(compare1, compare2, axis = 1)
comp = pd.DataFrame(comp)

compare3 = comp
#normalizing 0-1
from sklearn import preprocessing
x = compare3.values
MM = preprocessing.MinMaxScaler()
xScaled = MM.fit_transform(x)
compare3 = xScaled
compare3 = pd.DataFrame(compare3, columns=["a", "poisson", "c", "custom"])
compare3 = compare3.drop("c", axis = 1)
    
def DevianceLoss(npTrue, npPredict, dur): #np arrays
    dev = 0
    for i in range(0, len(npTrue)):
        yt = npTrue[i, 0]
        yp = npPredict[i, 0]
        d = dur[i, 0]
        logt = 0
        logp = 0
        
        if yt != 0:
            logt = yt * log(yt) 
        if yp !=0:
            logp = yt * log(yp)
        
        devTemp = 2*d*(logt - logp - yt + yp)
        dev = dev + devTemp
    
    return dev

devRMSE = DevianceLoss(y, preds1, d)
devCust = DevianceLoss(y, preds2, d)


