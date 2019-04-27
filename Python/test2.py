import tensorflow as tf
import keras 
import numpy as np 
import pandas as pd
from keras.layers import Dense
import keras.backend as KB
import math
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/upsample.csv")

factors = data.drop('NumberClaims', axis = 1)
factors = factors.drop('ClaimFrequency', axis = 1)
nbClaims = data['NumberClaims']
nbClaims = nbClaims.to_frame()
y = data['ClaimFrequency']
d = data['Duration']
y = y.to_frame()
d = d.values

def rmse(yt, yp):
    rmse = sqrt(mean_squared_error(yt, yp))
    return rmse
    
def custom_loss(data, y_pred):

    y_true = data[:, 0]
    d = data[:, 1]
    # condition
    y_true = max(y_true, 0.00001)
    mask2 = keras.backend.greater(y_true, 0)
    mask2 = KB.cast(mask2, 'float32')

    # returns 0 when y_true =0, 1 otherwise
    #calculate loss using d...
    loss_value = (1-mask2) * 2 * d * y_pred + mask2 * (2 * d * (y_true * KB.log(y_true) - y_true * KB.log(y_pred + KB.epsilon()) - y_true + y_pred))
    return loss_value

    
def custom_loss2(data, y_pred):

    y_true = data[:, 0]
    d = data[:, 1]
    # condition
    def f1(d, y_pred): return 2 * d * y_pred
    def f2(d, y_true, y_pred): return 2 * d * (y_true * KB.log(y_true) - y_true * KB.log(y_pred) - y_true + y_pred)

    loss_value = tf.cond(KB.greater(y_true, 0), f1(d, y_pred), f2(d, y_true, y_pred))
    
    return loss_value

def custom_loss3(data, y_pred):

    y_true = data[:, 0]
    d = data[:, 1]
    switch = KB.greater(y_true, 0.)
    # condition
    loss_value = KB.switch(switch, 2 * d * (y_true * KB.log(y_true) - y_true * KB.log(y_pred + KB.epsilon()) - y_true + y_pred), 2 * d * y_pred)
    return loss_value

def lf1(data, y_pred):
    y_true = data[:, 0]
    d = data[:, 1]
    return(2 * d * y_pred)

def lf2(data, y_pred):
    y_true = data[:, 0]
    d = data[:, 1]
    return(2 * d * (y_true * KB.log(y_true) - y_true * KB.log(y_pred) - y_true + y_pred))



def cl4(d):
    def custom_loss4(y_true, y_pred):
        if(d == 0):
            return lf1(y_true, y_pred)
        else:
            return lf2(y_true, y_pred)
    return custom_loss4
        



def baseline_model(loss):
    # create model
    #building model
    model = keras.Sequential()
    model.add(Dense(10, input_dim = 26, activation = "exponential"))
    model.add(Dense(5, input_dim = 26, activation = "exponential"))

    #model.add(Dense(10, activation = "relu"))
    model.add(Dense(1, activation = "exponential"))
    model.compile(loss=loss, optimizer='Adam')
    return model

model1 = baseline_model("poisson")
model2 = baseline_model(cl4(d))

model2.fit(factors, np.append(y, d, axis = 1), epochs=2, shuffle=True, verbose=1)
model1.fit(factors, y, epochs=2, shuffle=True, verbose=1)

#model 1 = poisson, model 2 = custom
#In spyder IDE can check comp to see if two models are similar
import math 
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





vect = np.vectorize(math.exp)
test = vect(preds)
test = pd.DataFrame(test)

rmse = rmse(y, preds)
test[127670:127680]
preds[127670:127680]
preds[3320:3330]
preds[69990:70000]
preds[100000:100020]
preds[0:20]


