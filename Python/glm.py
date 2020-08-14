import statsmodels.api as sm
import pandas as pd
import numpy as np

###############################
# Importing datasets
###############################
root = 'c:/users/kryst/desktop/poisson/poisson-neural-network-insurance-pricing'

import sys
sys.path.append(root + '/Python')
from adaboostr2class import AdaBoost

dataTrain = pd.read_csv(root + "/dataTrain.csv")

#separing columns
d1 = dataTrain['Duration']

y1 = dataTrain['ClaimFrequency']

nc1 = dataTrain['NumberClaims']

#importing test 
dataTest = pd.read_csv(root + "/dataTest.csv")

d1test = dataTest['Duration']

y1test = dataTest['ClaimFrequency']

#dropping useless dimensions
dataTrain = dataTrain.drop(columns=["Duration", "NumberClaims", "ClaimFrequency", "Unnamed: 0"])
dataTest = dataTest.drop(columns=["Duration", "NumberClaims", "ClaimFrequency", "Unnamed: 0"])

#Passing the Duration into keras is impossible cause there is two arguments only when creating a custom loss function.
#Therefore we use a trick and pass a tuple with duration and y instead. 
y1 = pd.DataFrame(y1)
d1 = pd.DataFrame(d1)

y1test = pd.DataFrame(y1test)
d1test = pd.DataFrame(d1test)

feed = np.append(y1, d1, axis = 1)
feed = pd.DataFrame(feed)

X = dataTrain
Xt = dataTest
y = y1
yt = y1test

X = sm.add_constant(X)
Xt = sm.add_constant(Xt)

#the stepwise was performed in R before being ported in python
Xbis = X[["Gender.F", "Zone.1", "Zone.2", "Zone.3", "Class.3", "Class.4", "BonusClass.4", "OwnersAge", "VehiculeAge"]]
Xtbis = Xt[["Gender.F", "Zone.1", "Zone.2", "Zone.3", "Class.3", "Class.4", "BonusClass.4", "OwnersAge", "VehiculeAge"]]

Xbis = sm.add_constant(Xbis)
Xtbis = sm.add_constant(Xtbis)

glm = sm.GLM(y, X, family = sm.families.Poisson(link = sm.families.links.log), duration = d1)    
glm = glm.fit()    
glm.summary()    

preds = glm.predict(Xt)
adb = AdaBoost()
devtot = adb.devFull(yt, preds, d1test)
devtot = devtot / 12901
devtot = devtot * 64501

glm2 = sm.GLM(y, Xbis, family = sm.families.Poisson(link=sm.families.links.log), duration = d1)
glm2 = glm2.fit()

preds2 = glm2.predict(Xtbis)
devtot2 = adb.devFull(yt, preds2, d1test)/12901*64501




