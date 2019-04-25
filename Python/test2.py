import tensorflow as tf
import keras 
import numpy as np 
import pandas as pd
from keras.layers import Dense
import keras.backend as KB
import math

data = pd.read_csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/samp.csv")

factors = data.drop('NumberClaims', axis = 1)
y = data['NumberClaims']
d = data['Duration']
y = y.to_frame()
d = d.to_frame()
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
    # condition
    loss_value = KB.switch(KB.greater(y_true, 0), 2 * d * y_pred, 2 * d * (y_true * KB.log(y_true + KB.epsilon()) - y_true * KB.log(y_pred + KB.epsilon()) - y_true + y_pred))
    return loss_value

def baseline_model():
    # create model
    #building model
    model = keras.Sequential()
    model.add(Dense(5, input_dim = 26, activation = "relu"))
    #model.add(Dense(10, activation = "relu"))
    model.add(Dense(1, activation = "exponential"))
    model.compile(loss=custom_loss3, optimizer='Adam')
    return model

model = baseline_model()
model.fit(factors, np.append(y, d, axis = 1), epochs=2, shuffle=True, verbose=1)
model.fit(factors, y, epochs=2, shuffle=True, verbose=1)


import math 
preds = model.predict(factors)

vect = np.vectorize(math.exp)
test = vect(preds)
test = pd.DataFrame(test)

test[127670:127680]
preds[127670:127680]
preds[3320:3330]
preds[69990:70000]
preds[100000:100020]
preds[0:20]
