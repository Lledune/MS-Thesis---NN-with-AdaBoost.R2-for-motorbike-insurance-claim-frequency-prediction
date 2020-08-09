import numpy as np
import pandas as pd
import os
os.getcwd()


#########################################
# Importing the data
#########################################
root = 'c:/users/kryst/desktop/poisson/poisson-neural-network-insurance-pricing'


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

########################################################
# Importing functions
########################################################
#Loss function     
from math import log

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
    
    
import statsmodels.api as sm 
X = dataTrain[['Gender.F', 'Zone.1', 'Zone.3', 'Class.3', 'Class.4', 'Class.6', 'BonusClass.4', 'OwnersAge', 'VehiculeAge']].copy()
Xtest = dataTest[['Gender.F', 'Zone.1', 'Zone.3', 'Class.3', 'Class.4', 'Class.6', 'BonusClass.4', 'OwnersAge', 'VehiculeAge']].copy()

y = y1
X = sm.add_constant(X)
Xtest = sm.add_constant(Xtest)

mod = sm.GLM(y, X, family = sm.families.Poisson(sm.families.links.log), duration = d1)
res = mod.fit()
preds = res.predict(Xtest)
dev = devFull(y1test, preds, d1test)
devtot = (dev/len(y1test))*64501






    