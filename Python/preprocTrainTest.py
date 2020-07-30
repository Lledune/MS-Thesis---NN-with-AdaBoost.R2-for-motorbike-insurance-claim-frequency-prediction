#dividing test and train sets for both used datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/preprocFull.csv")
datacats = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/catOnly.csv")

#shuffling because the dataset is ordered by age and the young clients have more accidents which leads to unbalanced k-fold.
#data = data.sample(frac = 1, random_state = 24202).reset_index(drop=True)
#datacats = datacats.sample(frac = 1, random_state = 24202).reset_index(drop=True)

#Create booleans for stratified division
data['bool'] = data['ClaimFrequency'] > 0
data['bool'] = data['bool'].astype(str)
datacats['bool'] = datacats['NumberClaims'] > 0
datacats['bool'] = datacats['bool'].astype(str)



dataTrain, dataTest, datacatsTrain, datacatsTest = train_test_split(data, datacats, random_state=24202,
                                                                           test_size = 0.125, stratify=data['bool'])

dataTest = dataTest.drop('bool', axis = 1)
dataTrain = dataTrain.drop('bool', axis = 1)
datacatsTest = datacatsTest.drop('bool', axis = 1)
datacatsTrain = datacatsTrain.drop('bool', axis = 1)
datacatsTest = datacatsTest.drop('Unnamed: 0', axis = 1)
datacatsTrain = datacatsTrain.drop('Unnamed: 0', axis = 1)

#Exporting CSV
dataTest.to_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/dataTest.csv")
dataTrain.to_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/dataTrain.csv")
datacatsTest.to_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/dataCatsTest.csv")
datacatsTrain.to_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/dataCatsTrain.csv")

