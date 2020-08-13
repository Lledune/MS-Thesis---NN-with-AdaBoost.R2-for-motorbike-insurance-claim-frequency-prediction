import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as KB
from math import log
from keras.layers import Dense, Dropout
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import _num_samples
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
import sys
import pickle
import statsmodels.api as sm 

#########################################
# Importing the data
#########################################
root = 'c:/users/kryst/desktop/poisson/poisson-neural-network-insurance-pricing'

sys.path.append(root + '/Python')
from AdaBoostR2Class import AdaBoost
from AdaBoostR2Class import AdaBoostBeta

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

################################################################
# Comparisons of the models + plots
################################################################

#########################################
# Functions
#########################################

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

#############################
# Loading models
#############################

#GLM
X = dataTrain
Xtest = dataTest
y = y1
X = sm.add_constant(X)
Xtest = sm.add_constant(Xtest)
impGLM = sm.load(root + '/Python/glm')

#NN
impNN = keras.models.load_model(root + '/Python/Models/NNmodel', custom_objects={'deviance' : deviance})

#DNN
impDNN = keras.models.load_model(root + '/Python/Models/DNNmodel', custom_objects={'deviance' : deviance})

#ADB
impADB = AdaBoost()
impADB.load_model(root + '/Python/Models/AdaBoost')



#####################################################
# Mean deviance for classes
#####################################################

#Creating dataset with appended prediction of cf
dataTestPred = dataTest.copy()
dataTestPred['predNN'] = impNN.predict(dataTest) 
dataTestPred['predDNN'] = impDNN.predict(dataTest)
dataTestPred['predGLM'] = impGLM.predict(Xtest)
dataTestPred['predADB'] = impADB.predict(dataTest)
dataTestPred['cf'] = y1test

####################
#man vs woman
####################

maskMan = (dataTestPred['Gender.F'] == 0)
men = dataTestPred[maskMan]
women = dataTestPred[~maskMan] 

#these array will be storing mean CF, for GLM, NN, DNN, ADB in that order.
menCF = []
womenCF = []
labels = ['GLM', 'NN', 'DNN', 'ADB', 'Realité']

menCF.append(np.mean(men['predGLM']))
menCF.append(np.mean(men['predNN']))
menCF.append(np.mean(men['predDNN']))
menCF.append(np.mean(men['predADB']))
menCF.append(np.mean(men['cf']))


womenCF.append(np.mean(women['predGLM']))
womenCF.append(np.mean(women['predNN']))
womenCF.append(np.mean(women['predDNN']))
womenCF.append(np.mean(women['predADB']))
womenCF.append(np.mean(women['cf']))

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x-width/2, menCF, width, label = 'Homme')
rects2 = ax.bar(x+width/2, womenCF, width, label = 'Femme')
ax.set_ylabel('Fréquence des sinistres moyenne')
ax.set_title('Fréquence des sinistres moyennes hommes/femmes')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

####################
#zone
####################

zone1CF = []
zone2CF = []
zone3CF = []
zone4CF = []
zone5CF = []
zone6CF = []
zone7CF = []

mask1 = (dataTestPred['Zone.1'] == 1)
mask2 = (dataTestPred['Zone.2'] == 1)
mask3 = (dataTestPred['Zone.3'] == 1)
mask4 = (dataTestPred['Zone.4'] == 1)
mask5 = (dataTestPred['Zone.5'] == 1)
mask6 = (dataTestPred['Zone.6'] == 1)
mask7 = ((mask1 == 0) & (mask2 == 0) & (mask3 == 0) & (mask4 == 0) & (mask5 == 0) & (mask6 == 0))

zone1 = dataTestPred[mask1]
zone2 = dataTestPred[mask2]
zone3 = dataTestPred[mask3]
zone4 = dataTestPred[mask4]
zone5 = dataTestPred[mask5]
zone6 = dataTestPred[mask6]
zone7 = dataTestPred[mask7]

zone1CF.append(np.mean(zone1['predGLM']))
zone2CF.append(np.mean(zone2['predGLM']))
zone3CF.append(np.mean(zone3['predGLM']))
zone4CF.append(np.mean(zone4['predGLM']))
zone5CF.append(np.mean(zone5['predGLM']))
zone6CF.append(np.mean(zone6['predGLM']))
zone7CF.append(np.mean(zone7['predGLM']))

zone1CF.append(np.mean(zone1['predNN']))
zone2CF.append(np.mean(zone2['predNN']))
zone3CF.append(np.mean(zone3['predNN']))
zone4CF.append(np.mean(zone4['predNN']))
zone5CF.append(np.mean(zone5['predNN']))
zone6CF.append(np.mean(zone6['predNN']))
zone7CF.append(np.mean(zone7['predNN']))

zone1CF.append(np.mean(zone1['predDNN']))
zone2CF.append(np.mean(zone2['predDNN']))
zone3CF.append(np.mean(zone3['predDNN']))
zone4CF.append(np.mean(zone4['predDNN']))
zone5CF.append(np.mean(zone5['predDNN']))
zone6CF.append(np.mean(zone6['predDNN']))
zone7CF.append(np.mean(zone7['predDNN']))

zone1CF.append(np.mean(zone1['predADB']))
zone2CF.append(np.mean(zone2['predADB']))
zone3CF.append(np.mean(zone3['predADB']))
zone4CF.append(np.mean(zone4['predADB']))
zone5CF.append(np.mean(zone5['predADB']))
zone6CF.append(np.mean(zone6['predADB']))
zone7CF.append(np.mean(zone7['predADB']))

zone1CF.append(np.mean(zone1['cf']))
zone2CF.append(np.mean(zone2['cf']))
zone3CF.append(np.mean(zone3['cf']))
zone4CF.append(np.mean(zone4['cf']))
zone5CF.append(np.mean(zone5['cf']))
zone6CF.append(np.mean(zone6['cf']))
zone7CF.append(np.mean(zone7['cf']))

x = np.arange(len(labels))
width = 0.1
fig, ax = plt.subplots()
rects1 = ax.bar(x-3*width, zone1CF, width, label = 'Banlieues')
rects2 = ax.bar(x-2*width, zone2CF, width, label = 'Petites villes')
rects3 = ax.bar(x-1*width, zone3CF, width, label = 'Villages')
rects4 = ax.bar(x-0*(width+(width/2)), zone4CF, width, label = 'Villes du Nord')
rects5 = ax.bar(x+1*width, zone5CF, width, label = 'Campagnes du Nord')
rects6 = ax.bar(x+2*width, zone6CF, width, label = 'Gotland')
rects7 = ax.bar(x+3*width, zone7CF, width, label = 'Parties centrales')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylabel('Fréquence des sinistres moyenne')
ax.set_title('Fréquence des sinistres moyenne par Zone')
fig.tight_layout()
plt.show()

##################
# class
##################

class1CF = []
class2CF = []
class3CF = []
class4CF = []
class5CF = []
class6CF = []
class0CF = []

mask1 = (dataTestPred['Class.1'] == 1)
mask2 = (dataTestPred['Class.2'] == 1)
mask3 = (dataTestPred['Class.3'] == 1)
mask4 = (dataTestPred['Class.4'] == 1)
mask5 = (dataTestPred['Class.5'] == 1)
mask6 = (dataTestPred['Class.6'] == 1)
mask0 = ((mask1 == 0) & (mask2 == 0) & (mask3 == 0) & (mask4 == 0) & (mask5 == 0) & (mask6 == 0))

class1 = dataTestPred[mask1]
class2 = dataTestPred[mask2]
class3 = dataTestPred[mask3]
class4 = dataTestPred[mask4]
class5 = dataTestPred[mask5]
class6 = dataTestPred[mask6]
class0 = dataTestPred[mask0]

class1CF.append(np.mean(class1['predGLM']))
class2CF.append(np.mean(class2['predGLM']))
class3CF.append(np.mean(class3['predGLM']))
class4CF.append(np.mean(class4['predGLM']))
class5CF.append(np.mean(class5['predGLM']))
class6CF.append(np.mean(class6['predGLM']))
class0CF.append(np.mean(class0['predGLM']))

class1CF.append(np.mean(class1['predNN']))
class2CF.append(np.mean(class2['predNN']))
class3CF.append(np.mean(class3['predNN']))
class4CF.append(np.mean(class4['predNN']))
class5CF.append(np.mean(class5['predNN']))
class6CF.append(np.mean(class6['predNN']))
class0CF.append(np.mean(class0['predNN']))

class1CF.append(np.mean(class1['predDNN']))
class2CF.append(np.mean(class2['predDNN']))
class3CF.append(np.mean(class3['predDNN']))
class4CF.append(np.mean(class4['predDNN']))
class5CF.append(np.mean(class5['predDNN']))
class6CF.append(np.mean(class6['predDNN']))
class0CF.append(np.mean(class0['predDNN']))

class1CF.append(np.mean(class1['predADB']))
class2CF.append(np.mean(class2['predADB']))
class3CF.append(np.mean(class3['predADB']))
class4CF.append(np.mean(class4['predADB']))
class5CF.append(np.mean(class5['predADB']))
class6CF.append(np.mean(class6['predADB']))
class0CF.append(np.mean(class0['predADB']))

class1CF.append(np.mean(class1['cf']))
class2CF.append(np.mean(class2['cf']))
class3CF.append(np.mean(class3['cf']))
class4CF.append(np.mean(class4['cf']))
class5CF.append(np.mean(class5['cf']))
class6CF.append(np.mean(class6['cf']))
class0CF.append(np.mean(class0['cf']))

x = np.arange(len(labels))
width = 0.1
fig, ax = plt.subplots()
rects0 = ax.bar(x-3*width, class0CF, width, label = 'Classe 0')
rects1 = ax.bar(x-2*width, class1CF, width, label = 'Classe 1')
rects2 = ax.bar(x-1*width, class2CF, width, label = 'Classe 2')
rects3 = ax.bar(x-0*width, class3CF, width, label = 'Classe 3')
rects4 = ax.bar(x+1*width, class4CF, width, label = 'Classe 4')
rects5 = ax.bar(x+2*width, class5CF, width, label = 'Classe 5')
rects6 = ax.bar(x+3*width, class6CF, width, label = 'Classe 6')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylabel('Fréquence des sinistres moyenne')
ax.set_title('Fréquence des sinistres moyenne par classe de véhicule')
fig.tight_layout()
plt.show()

##################
# Bonusclass
##################

Bonusclass1CF = []
Bonusclass2CF = []
Bonusclass3CF = []
Bonusclass4CF = []
Bonusclass5CF = []
Bonusclass6CF = []
Bonusclass0CF = []

mask1 = (dataTestPred['BonusClass.1'] == 1)
mask2 = (dataTestPred['BonusClass.2'] == 1)
mask3 = (dataTestPred['BonusClass.3'] == 1)
mask4 = (dataTestPred['BonusClass.4'] == 1)
mask5 = (dataTestPred['BonusClass.5'] == 1)
mask6 = (dataTestPred['BonusClass.6'] == 1)
mask0 = ((mask1 == 0) & (mask2 == 0) & (mask3 == 0) & (mask4 == 0) & (mask5 == 0) & (mask6 == 0))

Bonusclass1 = dataTestPred[mask1]
Bonusclass2 = dataTestPred[mask2]
Bonusclass3 = dataTestPred[mask3]
Bonusclass4 = dataTestPred[mask4]
Bonusclass5 = dataTestPred[mask5]
Bonusclass6 = dataTestPred[mask6]
Bonusclass0 = dataTestPred[mask0]

Bonusclass1CF.append(np.mean(Bonusclass1['predGLM']))
Bonusclass2CF.append(np.mean(Bonusclass2['predGLM']))
Bonusclass3CF.append(np.mean(Bonusclass3['predGLM']))
Bonusclass4CF.append(np.mean(Bonusclass4['predGLM']))
Bonusclass5CF.append(np.mean(Bonusclass5['predGLM']))
Bonusclass6CF.append(np.mean(Bonusclass6['predGLM']))
Bonusclass0CF.append(np.mean(Bonusclass0['predGLM']))

Bonusclass1CF.append(np.mean(Bonusclass1['predNN']))
Bonusclass2CF.append(np.mean(Bonusclass2['predNN']))
Bonusclass3CF.append(np.mean(Bonusclass3['predNN']))
Bonusclass4CF.append(np.mean(Bonusclass4['predNN']))
Bonusclass5CF.append(np.mean(Bonusclass5['predNN']))
Bonusclass6CF.append(np.mean(Bonusclass6['predNN']))
Bonusclass0CF.append(np.mean(Bonusclass0['predNN']))

Bonusclass1CF.append(np.mean(Bonusclass1['predDNN']))
Bonusclass2CF.append(np.mean(Bonusclass2['predDNN']))
Bonusclass3CF.append(np.mean(Bonusclass3['predDNN']))
Bonusclass4CF.append(np.mean(Bonusclass4['predDNN']))
Bonusclass5CF.append(np.mean(Bonusclass5['predDNN']))
Bonusclass6CF.append(np.mean(Bonusclass6['predDNN']))
Bonusclass0CF.append(np.mean(Bonusclass0['predDNN']))

Bonusclass1CF.append(np.mean(Bonusclass1['predADB']))
Bonusclass2CF.append(np.mean(Bonusclass2['predADB']))
Bonusclass3CF.append(np.mean(Bonusclass3['predADB']))
Bonusclass4CF.append(np.mean(Bonusclass4['predADB']))
Bonusclass5CF.append(np.mean(Bonusclass5['predADB']))
Bonusclass6CF.append(np.mean(Bonusclass6['predADB']))
Bonusclass0CF.append(np.mean(Bonusclass0['predADB']))

Bonusclass1CF.append(np.mean(Bonusclass1['cf']))
Bonusclass2CF.append(np.mean(Bonusclass2['cf']))
Bonusclass3CF.append(np.mean(Bonusclass3['cf']))
Bonusclass4CF.append(np.mean(Bonusclass4['cf']))
Bonusclass5CF.append(np.mean(Bonusclass5['cf']))
Bonusclass6CF.append(np.mean(Bonusclass6['cf']))
Bonusclass0CF.append(np.mean(Bonusclass0['cf']))

x = np.arange(len(labels))
width = 0.1
fig, ax = plt.subplots()
rects0 = ax.bar(x-3*width, Bonusclass0CF, width, label = 'Classe de bonus 0')
rects1 = ax.bar(x-2*width, Bonusclass1CF, width, label = 'Classe de bonus 1')
rects2 = ax.bar(x-1*width, Bonusclass2CF, width, label = 'Classe de bonus 2')
rects3 = ax.bar(x-0*width, Bonusclass3CF, width, label = 'Classe de bonus 3')
rects4 = ax.bar(x+1*width, Bonusclass4CF, width, label = 'Classe de bonus 4')
rects5 = ax.bar(x+2*width, Bonusclass5CF, width, label = 'Classe de bonus 5')
rects6 = ax.bar(x+3*width, Bonusclass6CF, width, label = 'Classe de bonus 6')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylabel('Fréquence des sinistres moyenne')
ax.set_title('Fréquence des sinistres moyenne par classe de bonus')
fig.tight_layout()
plt.show()

##############
#ownerAge
##############

#un-normalize
dataTestPred['OwnersAge'] = dataTestPred['OwnersAge'] * 92

age0CF = []
age1CF = []
age2CF = []
age3CF = []
age4CF = []
age5CF = []
age6CF = []

higherBounds = {
    0 : 24,
    1 : 34,
    2 : 44,
    3 : 54,
    4 : 64,
    5 : 74,
    }

#masks
mask0 = (dataTestPred['OwnersAge'] <= higherBounds[0])  
mask1 = (dataTestPred['OwnersAge'] <= higherBounds[1]) & (dataTestPred['OwnersAge'] > higherBounds[0])
mask2 = (dataTestPred['OwnersAge'] <= higherBounds[2]) & (dataTestPred['OwnersAge'] > higherBounds[1])
mask3 = (dataTestPred['OwnersAge'] <= higherBounds[3]) & (dataTestPred['OwnersAge'] > higherBounds[2])
mask4 = (dataTestPred['OwnersAge'] <= higherBounds[4]) & (dataTestPred['OwnersAge'] > higherBounds[3])
mask5 = (dataTestPred['OwnersAge'] <= higherBounds[5]) & (dataTestPred['OwnersAge'] > higherBounds[4])
mask6 = (dataTestPred['OwnersAge'] > higherBounds[5])

age0 = dataTestPred[mask0]
age1 = dataTestPred[mask1]
age2 = dataTestPred[mask2]
age3 = dataTestPred[mask3]
age4 = dataTestPred[mask4]
age5 = dataTestPred[mask5]
age6 = dataTestPred[mask6]

age0CF.append(np.mean(age0['predGLM']))
age1CF.append(np.mean(age1['predGLM']))
age2CF.append(np.mean(age2['predGLM']))
age3CF.append(np.mean(age3['predGLM']))
age4CF.append(np.mean(age4['predGLM']))
age5CF.append(np.mean(age5['predGLM']))
age6CF.append(np.mean(age6['predGLM']))

age0CF.append(np.mean(age0['predNN']))
age1CF.append(np.mean(age1['predNN']))
age2CF.append(np.mean(age2['predNN']))
age3CF.append(np.mean(age3['predNN']))
age4CF.append(np.mean(age4['predNN']))
age5CF.append(np.mean(age5['predNN']))
age6CF.append(np.mean(age6['predNN']))

age0CF.append(np.mean(age0['predDNN']))
age1CF.append(np.mean(age1['predDNN']))
age2CF.append(np.mean(age2['predDNN']))
age3CF.append(np.mean(age3['predDNN']))
age4CF.append(np.mean(age4['predDNN']))
age5CF.append(np.mean(age5['predDNN']))
age6CF.append(np.mean(age6['predDNN']))

age0CF.append(np.mean(age0['predADB']))
age1CF.append(np.mean(age1['predADB']))
age2CF.append(np.mean(age2['predADB']))
age3CF.append(np.mean(age3['predADB']))
age4CF.append(np.mean(age4['predADB']))
age5CF.append(np.mean(age5['predADB']))
age6CF.append(np.mean(age6['predADB']))

age0CF.append(np.mean(age0['cf']))
age1CF.append(np.mean(age1['cf']))
age2CF.append(np.mean(age2['cf']))
age3CF.append(np.mean(age3['cf']))
age4CF.append(np.mean(age4['cf']))
age5CF.append(np.mean(age5['cf']))
age6CF.append(np.mean(age6['cf']))

x = np.arange(len(labels))
width = 0.1
fig, ax = plt.subplots()
rects0 = ax.bar(x-3*width, age0CF, width, label = '16-24')
rects1 = ax.bar(x-2*width, age1CF, width, label = '25-34')
rects2 = ax.bar(x-1*width, age2CF, width, label = '35-44')
rects3 = ax.bar(x-0*width, age3CF, width, label = '45-54')
rects4 = ax.bar(x+1*width, age4CF, width, label = '55-64')
rects5 = ax.bar(x+2*width, age5CF, width, label = '65-74')
rects6 = ax.bar(x+3*width, age6CF, width, label = '74+')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylabel('Fréquence des sinistres moyenne')
ax.set_title('Fréquence des sinistres moyenne par age')
fig.tight_layout()
plt.show()

#make bar graph of claim frequ totals for each model and reality
claimfreqs = []

claimfreqs.append(np.sum(dataTestPred['predGLM']))
claimfreqs.append(np.sum(dataTestPred['predNN']))
claimfreqs.append(np.sum(dataTestPred['predDNN']))
claimfreqs.append(np.sum(dataTestPred['predADB']))
claimfreqs.append(np.sum(dataTestPred['cf']))

x = np.arange(len(labels))
width = 0.4
fig, ax = plt.subplots()
rects0 = ax.bar(x-0*width, claimfreqs, width, label = 'Fréquence des sinistres totale')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylabel('Fréquence des sinistres totale')
ax.set_title('Fréquence des sinistres moyenne par modèle')
fig.tight_layout()
plt.show()

#################################################################################################
# Création d'assurés types
#################################################################################################

#Medianes ? 
datattt = pd.read_csv(root + '/mcc.csv')
np.median(datattt['VehiculeAge']) #12
#Zone : 4, Bonus : 7, Class : 3
#max of vehiculeage is 83 (for normalization, without the removed outlier)
#max of driver age is 92

#Homme, 20 ans, village
h20 = np.array([[0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.2174,0.1445]])
h20glm = [1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.2174,0.1445]

#Homme, 50 ans, village
h50 = np.array([[0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.6024,0.1445]])
h50glm = [1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.6024,0.1445]

#Femme, 20 ans, village
f20 = np.array([[1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.2174,0.1445]])
f20glm = [1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.2174,0.1445]

#Femme, 50 ans, village
f50 = np.array([[1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.6024,0.1445]])
f50glm = [1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.6024,0.1445]

#Homme, 20 ans, grande ville
h20V = np.array([[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.2174,0.1445]])
h20glmV = [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.2174,0.1445]

#Homme, 50 ans, grande ville
h50V = np.array([[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.6024,0.1445]])
h50glmV = [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.6024,0.1445]

#Femme, 20 ans, grande ville
f20V = np.array([[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.2174,0.1445]])
f20glmV = [1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.2174,0.1445]

#Femme, 50 ans, grande ville
f50V = np.array([[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.6024,0.1445]])
f50glmV = [1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0.6024,0.1445]

##############
# Prédictions village
##############

#Age man

temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []
predADBdev = []

ranged = range(1,93)

for i in range(1, 93): 
    temp2 = h20.copy()
    temp2glm = h20glm.copy()
    
    #change age
    temp2[0][19] = i/92.
    temp2glm[20] = i/92.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Homme, village')
plt.xlabel('Age')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#Age woman

temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,93)

for i in range(1, 93): 
    temp2 = f20.copy()
    temp2glm = f20glm.copy()
    
    #change age
    temp2[0][19] = i/92.
    temp2glm[20] = i/92.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Femme, village')
plt.xlabel('Age')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

##################
#VehiculeAge
##################

#man, 20
temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,84)

for i in range(1, 84): 
    temp2 = h20.copy()
    temp2glm = h20glm.copy()
    
    #change age of vehicle
    temp2[0][20] = i/83.
    temp2glm[21] = i/83.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Homme, 20 ans, village')
plt.xlabel('Age du véhicule')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


#man, 50
temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,84)

for i in range(1, 84): 
    temp2 = h50.copy()
    temp2glm = h50glm.copy()
    
    #change age of vehicle
    temp2[0][20] = i/83.
    temp2glm[21] = i/83.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Homme, 50 ans, village')
plt.xlabel('Age du véhicule')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


#woman, 20
temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,84)

for i in range(1, 84): 
    temp2 = f20.copy()
    temp2glm = f20glm.copy()
    
    #change age of vehicle
    temp2[0][20] = i/83.
    temp2glm[21] = i/83.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Femme, 20 ans, village')
plt.xlabel('Age du véhicule')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#woman, 50
temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,84)

for i in range(1, 84): 
    temp2 = f50.copy()
    temp2glm = f50glm.copy()
    
    #change age of vehicle
    temp2[0][20] = i/83.
    temp2glm[21] = i/83.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Femme, 50 ans, village')
plt.xlabel('Age du véhicule')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


##############
# Prédictions grande ville
##############

#Age man

temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,93)

for i in range(1, 93): 
    temp2 = h20V.copy()
    temp2glm = h20glmV.copy()
    
    #change age
    temp2[0][19] = i/92.
    temp2glm[20] = i/92.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Homme, grande ville')
plt.xlabel('Age')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#Age woman

temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,93)

for i in range(1, 93): 
    temp2 = f20V.copy()
    temp2glm = f20glmV.copy()
    
    #change age
    temp2[0][19] = i/92.
    temp2glm[20] = i/92.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Femme, grande ville')
plt.xlabel('Age')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

##################
#VehiculeAge
##################

#man, 20
temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,84)

for i in range(1, 84): 
    temp2 = h20V.copy()
    temp2glm = h20glmV.copy()
    
    #change age of vehicle
    temp2[0][20] = i/83.
    temp2glm[21] = i/83.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Homme, 20 ans, grande ville')
plt.xlabel('Age du véhicule')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


#man, 50
temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,84)

for i in range(1, 84): 
    temp2 = h50V.copy()
    temp2glm = h50glmV.copy()
    
    #change age of vehicle
    temp2[0][20] = i/83.
    temp2glm[21] = i/83.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Homme, 50 ans, grande ville')
plt.xlabel('Age du véhicule')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


#woman, 20
temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,84)

for i in range(1, 84): 
    temp2 = f20V.copy()
    temp2glm = f20glmV.copy()
    
    #change age of vehicle
    temp2[0][20] = i/83.
    temp2glm[21] = i/83.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Femme, 20 ans, grande ville')
plt.xlabel('Age du véhicule')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#woman, 50
temp = [] #storing the data that will be used 
predGLM = []
predNN = []
predDNN = []
predADB = []

ranged = range(1,84)

for i in range(1, 84): 
    temp2 = f50V.copy()
    temp2glm = f50glmV.copy()
    
    #change age of vehicle
    temp2[0][20] = i/83.
    temp2glm[21] = i/83.

    temp.append(temp2)
    predGLM.append(impGLM.predict(temp2glm)[0])
    predNN.append(impNN.predict(temp2)[0])
    predDNN.append(impDNN.predict(temp2)[0])
    predADB.append(impADB.predict(temp2)[0])

plt.plot(ranged, predGLM, label = 'GLM')
plt.plot(ranged, predNN, label = 'NN')
plt.plot(ranged, predDNN, label = 'DNN')
plt.plot(ranged, predADB, label = 'ADB')
plt.title('Femme, 50 ans, grande ville')
plt.xlabel('Age du véhicule')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#######################################################
# Analyse de la déviance en fonction de la prédiction (graphe montrant l'évolution de la déviance pour y = 0 et y = 1)
#######################################################


###############################
# Plotting estimator errors
###############################

plt.plot(range(1,len(impADB.estimatorsErrors)+1), impADB.estimatorsErrors, linestyle = 'dashed', alpha = 0.7)
plt.title('Erreurs des prédicteurs ADB')
plt.show()
plt.close()


#different learning rates on client examples
#create multiple adaboost 
lrsad = [0.01, 0.1, 0.5, 1]
adbstore = []
#create static params
staticparamsListLR = []

for i in range(0, len(lrsad)):
    temp = {
            'n_est' : [500],
            'loss' : ['exponential'],
            'learning_rate' : [lrsad[i]],
            'kerasEpochs' : [300],
            'kerasBatchSize' : [51600],
            'dropout' : [0.1],
            'nn1' : [12],
            'keraslr' : [0.15],
        }
    staticparamsListLR.append(temp)


for i in range(0, len(lrsad)):
    print(i)
    params = staticparamsListLR[i]
    estimator = AdaBoost(n_est=params['n_est'][0], loss = params['loss'][0], learning_rate=params['learning_rate'][0], kerasEpochs=params['kerasEpochs'][0],
                         kerasBatchSize=params['kerasBatchSize'][0], dropout = params['dropout'][0], nn1=params['nn1'][0], keraslr=params['keraslr'][0], 
                         input_dim=21)
    
    initw = estimator.initWeights(dataTrain)
    estimator.fit(dataTrain, feed, initw)
    adbstore.append(estimator)    
adb1 = adbstore[0]
adb2 = adbstore[1]
adb3 = adbstore[2]
adb4 = adbstore[3]

#Homme, village

temp = [] #storing the data that will be used 
predadb1 = []
predadb2 = []
predadb3 = []
predadb4 = []

ranged = range(1,93)

for i in range(1, 93): 
    temp2 = h20.copy()
    print(i)
    
    #change age
    temp2[0][19] = i/92.

    temp.append(temp2)
    predadb1.append(adb1.predict(temp2)[0])
    predadb2.append(adb2.predict(temp2)[0])
    predadb3.append(adb3.predict(temp2)[0])
    predadb4.append(adb4.predict(temp2)[0])

plt.plot(ranged, predadb1, label = 'LR = 0.01')
plt.plot(ranged, predadb2, label = 'LR = 0.1')
plt.plot(ranged, predadb3, label = 'LR = 0.5')
plt.plot(ranged, predadb4, label = 'LR = 1')
plt.title('Homme, village')
plt.xlabel('Age')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#Homme, ville

temp = [] #storing the data that will be used 
predadb1 = []
predadb2 = []
predadb3 = []
predadb4 = []

ranged = range(1,93)

for i in range(1, 93): 
    temp2 = h20V.copy()
    print(i)
    
    #change age
    temp2[0][19] = i/92.

    temp.append(temp2)
    predadb1.append(adb1.predict(temp2)[0])
    predadb2.append(adb2.predict(temp2)[0])
    predadb3.append(adb3.predict(temp2)[0])
    predadb4.append(adb4.predict(temp2)[0])

plt.plot(ranged, predadb1, label = 'LR = 0.01')
plt.plot(ranged, predadb2, label = 'LR = 0.1')
plt.plot(ranged, predadb3, label = 'LR = 0.5')
plt.plot(ranged, predadb4, label = 'LR = 1')
plt.title('Homme, ville')
plt.xlabel('Age')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#femme, village

temp = [] #storing the data that will be used 
predadb1 = []
predadb2 = []
predadb3 = []
predadb4 = []

ranged = range(1,93)

for i in range(1, 93): 
    temp2 = f20.copy()
    print(i)
    
    #change age
    temp2[0][19] = i/92.

    temp.append(temp2)
    predadb1.append(adb1.predict(temp2)[0])
    predadb2.append(adb2.predict(temp2)[0])
    predadb3.append(adb3.predict(temp2)[0])
    predadb4.append(adb4.predict(temp2)[0])

plt.plot(ranged, predadb1, label = 'LR = 0.01')
plt.plot(ranged, predadb2, label = 'LR = 0.1')
plt.plot(ranged, predadb3, label = 'LR = 0.5')
plt.plot(ranged, predadb4, label = 'LR = 1')
plt.title('Femme, village')
plt.xlabel('Age')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#femme, ville
temp = [] #storing the data that will be used 
predadb1 = []
predadb2 = []
predadb3 = []
predadb4 = []

ranged = range(1,93)

for i in range(1, 93): 
    temp2 = f20V.copy()
    print(i)
    
    #change age
    temp2[0][19] = i/92.

    temp.append(temp2)
    predadb1.append(adb1.predict(temp2)[0])
    predadb2.append(adb2.predict(temp2)[0])
    predadb3.append(adb3.predict(temp2)[0])
    predadb4.append(adb4.predict(temp2)[0])

plt.plot(ranged, predadb1, label = 'LR = 0.01')
plt.plot(ranged, predadb2, label = 'LR = 0.1')
plt.plot(ranged, predadb3, label = 'LR = 0.5')
plt.plot(ranged, predadb4, label = 'LR = 1')
plt.title('Femme, ville')
plt.xlabel('Age')
plt.ylabel('Fréquence des sinistres prédite')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#dev ada
adb = AdaBoost(20,'deviance', 1, 200, 51600, 0.1, 15, 0.3, 21)
initWeights = adb.initWeights(dataTrain)
adb.fit(dataTrain, feed, initWeights)


estErrors = adb.estimatorsErrors
ranged = range(0, len(estErrors))
plt.plot(ranged[0:19], estErrors[0:19], alpha = 0.7, linestyle = 'dashed')
plt.title("Évolution de l'erreur moyenne (AdaBoost.R2 déviance)")
plt.xlabel('Itérations')
plt.ylabel('Erreur moyenne')


#Comparaison claim freq certains clients
#création jeu de données (4 clients)
glmtempdata = dataTest.copy()
glmtempdata = sm.add_constant(glmtempdata)

c1data = pd.DataFrame(dataTest.iloc[57]).T #cf 0.5, homme, villes du nord de la suède, 49 ans
c2data = pd.DataFrame(dataTest.iloc[233]).T #cf 0.7, homme, ville moyenne, 29 ans
c3data = pd.DataFrame(dataTest.iloc[254]).T #femme, campagne du nord, 43 ans
c4data = pd.DataFrame(dataTest.iloc[300]).T #homme, ville moyenne, 27 ans

c1dataglm = np.array(glmtempdata.iloc[57]) #cf 0.5
c2dataglm = np.array(glmtempdata.iloc[233]) #cf 0.7
c3dataglm = np.array(glmtempdata.iloc[254])
c4dataglm = np.array(glmtempdata.iloc[300])

c1y = y1test.iloc[57]
c2y = y1test.iloc[233]
c3y = y1test.iloc[250]
c4y = y1test.iloc[300]

#predictions
glmpreds = []
nnpreds = []
dnnpreds = []
adbpreds = []
reality = []

reality.append(c1y[0])
reality.append(c2y[0])
reality.append(c3y[0])
reality.append(c4y[0])

glmpreds.append(impGLM.predict(c1dataglm)[0])
glmpreds.append(impGLM.predict(c2dataglm)[0])
glmpreds.append(impGLM.predict(c3dataglm)[0])
glmpreds.append(impGLM.predict(c4dataglm)[0])

nnpreds.append(impNN.predict(c1data)[0][0])
nnpreds.append(impNN.predict(c2data)[0][0])
nnpreds.append(impNN.predict(c3data)[0][0])
nnpreds.append(impNN.predict(c4data)[0][0])

dnnpreds.append(impDNN.predict(c1data)[0][0])
dnnpreds.append(impDNN.predict(c2data)[0][0])
dnnpreds.append(impDNN.predict(c3data)[0][0])
dnnpreds.append(impDNN.predict(c4data)[0][0])

adbpreds.append(impADB.predict(c1data)[0])
adbpreds.append(impADB.predict(c2data)[0])
adbpreds.append(impADB.predict(c3data)[0])
adbpreds.append(impADB.predict(c4data)[0])


labels = ["H, 49 ans", "H, 29 ans", "F, 43 ans", "H, 27 ans"]


x = np.arange(len(labels))

width = 0.1
fig, ax = plt.subplots()
rects0 = ax.bar(x-2*width, glmpreds, width, label = 'GLM')
rects2 = ax.bar(x-1*width, nnpreds, width, label = 'NN')
rects3 = ax.bar(x-0*width, dnnpreds, width, label = 'DNN')
rects4 = ax.bar(x+1*width, adbpreds, width, label = 'ADB')
rects4 = ax.bar(x+2*width, reality, width, label = 'Reality')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylabel('Fréquence des sinistres prédite')
ax.set_title('Fréquence des sinistres prédites vs valeurs réelles')
fig.tight_layout()
plt.show()






    