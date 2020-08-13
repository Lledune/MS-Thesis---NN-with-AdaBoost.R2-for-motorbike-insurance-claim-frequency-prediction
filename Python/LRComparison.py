import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#########################################
#!!! change root to your main folder !!
#########################################
root = 'c:/users/kryst/desktop/poisson/poisson-neural-network-insurance-pricing'

sys.path.append(root + '/Python')
from AdaBoostR2Class import AdaBoost
from AdaBoostR2Class import AdaBoostBeta

###############################
# Importing datasets
###############################
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
feed = pd.DataFrame(feed)
feed2 = pd.DataFrame(feed2)
        
def getParams(ada = AdaBoostBeta()):
    """
    Returns a dictionary of parameters and evaluation metrics for a given AdaBoost object.
    """
    config_dic = {}
    config_dic['n_est'] = ada.n_est
    config_dic['loss'] = ada.loss
    config_dic['estimatorsErrors'] = ada.estimatorsErrors
    config_dic['estimatorsWeights'] = ada.estimatorsWeights
    config_dic['estimatorsSampleWeights'] = ada.estimatorsSampleWeights
    config_dic['learning_rate'] = ada.learning_rate
    config_dic['averageLoss'] = ada.averageLoss
    config_dic['kerasEpochs'] = ada.kerasEpochs
    config_dic['kerasBatchSize'] = ada.kerasBatchSize
    config_dic['dropout'] = ada.dropout
    config_dic['nn1'] = ada.nn1
    config_dic['keraslr'] = ada.keraslr
    config_dic['input_dim'] = ada.input_dim
    #adding the losses
    preds = ada.predict(dataTest)
    testLoss = ada.devFull(y1test, preds, d1test)
    meanTestLoss = testLoss/len(preds)
    fullTestLoss = meanTestLoss * 64501
    config_dic['testLoss'] = testLoss 
    config_dic['meanTestLoss'] = meanTestLoss
    config_dic['fullTestLoss'] = fullTestLoss
    config_dic['real_n_est'] = len(ada.estimators)
    return config_dic


#keep learning to 1 for beta version or it is not correct !!!
adbb = AdaBoostBeta(n_est=20, loss = 'exponential', learning_rate = 1, kerasEpochs = 150, kerasBatchSize=51600, dropout = 0.1, nn1 = 5,
                    keraslr = 0.5)
initWeights = adbb.initWeights(dataTrain)
adbb.fit(dataTrain, feed, initWeights)

adba1 = AdaBoost(n_est=20, loss = 'exponential', learning_rate = 1, kerasEpochs = 150, kerasBatchSize=51600, dropout = 0.1, nn1 = 5,
                    keraslr = 0.5)
adba1.fit(dataTrain, feed, initWeights)

adba2 = AdaBoost(n_est=20, loss = 'exponential', learning_rate = 0.1, kerasEpochs = 150, kerasBatchSize=51600, dropout = 0.1, nn1 = 5,
                    keraslr = 0.5)
adba2.fit(dataTrain, feed, initWeights)

adba3 = AdaBoost(n_est=20, loss = 'exponential', learning_rate = 0.5, kerasEpochs = 150, kerasBatchSize=51600, dropout = 0.1, nn1 = 5,
                    keraslr = 0.5)
adba3.fit(dataTrain, feed, initWeights)


#plotting evolution of estimator weights
weightsAlpha1 = adba1.estimatorsWeights
weightsAlpha2 = adba2.estimatorsWeights
weightsAlpha3 = adba3.estimatorsWeights
weightsBeta = adbb.estimatorsWeights

rangeAlpha1 = range(0, len(weightsAlpha1))
rangeAlpha2 = range(0, len(weightsAlpha2))
rangeAlpha3 = range(0, len(weightsAlpha3))
rangeBeta = range(0, len(weightsBeta))

plt.plot(rangeAlpha1, weightsAlpha1, label = 'LR = 1', alpha = 0.5)
plt.plot(rangeAlpha3, weightsAlpha3, label = 'LR = 0.5', alpha = 0.5)
plt.plot(rangeAlpha2, weightsAlpha2, label = 'LR = 0.1', alpha = 0.5)
plt.plot(rangeBeta, weightsBeta, label = 'Version originale', linestyle = 'dashed')
plt.title("Comparaison des différentes versions d'AdaBoost.R2")
plt.ylabel("Importance de l'apprenant faible")
plt.xlabel("Itérations")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#Plotting evolution of sample weight (n = 0)
weightsSampAlpha1 = adba1.estimatorsSampleWeights
weightsSampAlpha2 = adba2.estimatorsSampleWeights
weightsSampAlpha3 = adba3.estimatorsSampleWeights
weightsSampBeta = adbb.estimatorsSampleWeights

alpha1data = []
alpha2data = []
alpha3data = []
betadata = []

for vector in weightsSampAlpha1:
    alpha1data.append(vector[0])
    
for vector in weightsSampAlpha2:
    alpha2data.append(vector[0])

for vector in weightsSampAlpha3:
    alpha3data.append(vector[0])
    
for vector in weightsSampBeta:
    betadata.append(vector[0])
    
rangeAlpha1 = range(0, len(alpha1data))
rangeAlpha2 = range(0, len(alpha2data))
rangeAlpha3 = range(0, len(alpha3data))
rangeBeta = range(0, len(betadata))

    
    

plt.plot(rangeAlpha1, alpha1data, label = "LR = 1", alpha = 0.5)
plt.plot(rangeAlpha3, alpha3data, label = "LR = 0.5", alpha = 0.5)
plt.plot(rangeAlpha2, alpha2data, label = "LR = 0.1", alpha = 0.5)
plt.plot(rangeBeta, betadata, label = "Version originale", linestyle = 'dashed')
plt.title("Évolution d'un poids")
plt.ylabel("Valeur du poids")
plt.xlabel("Itérations")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


#plotting evolution of estimator weights
weightsAlpha1 = adba1.averageLoss
weightsAlpha2 = adba2.averageLoss
weightsAlpha3 = adba3.averageLoss
weightsBeta = adbb.averageLoss

rangeAlpha1 = range(0, len(weightsAlpha1))
rangeAlpha2 = range(0, len(weightsAlpha2))
rangeAlpha3 = range(0, len(weightsAlpha3))
rangeBeta = range(0, len(weightsBeta))

plt.plot(rangeAlpha1, weightsAlpha1, label = 'LR = 1', alpha = 0.5)
plt.plot(rangeAlpha3, weightsAlpha3, label = 'LR = 0.5', alpha = 0.5)
plt.plot(rangeAlpha2, weightsAlpha2, label = 'LR = 0.1', alpha = 0.5)
plt.plot(rangeBeta, weightsBeta, label = 'Version originale', linestyle = 'dashed')
plt.title("Comparaison des différentes versions d'AdaBoost.R2")
plt.ylabel("Erreur moyenne")
plt.xlabel("Itérations")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


