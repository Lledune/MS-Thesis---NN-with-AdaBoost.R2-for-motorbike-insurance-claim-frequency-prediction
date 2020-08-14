#This file was used to test if a dataset with intervals for continuous variables would be beneficial
#results wern't very good so it won't be used.

import numpy as np 
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#preprocessing 
data = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/mcc.csv")

#creating columns 
data['OwnersAgeCat'] = None
data['VehiculeAgeCat'] = None

#ownerAge
higherBounds = {
    0 : 24,
    1 : 34,
    2 : 44,
    3 : 54,
    4 : 64,
    5 : 74,
    6 : 84,
    7 : 94
    }

#vehicleAge
high = {
        0 : 9,
        1 : 19,
        2 : 29,
        3 : 39,
        4 : 49,
        }

#masks
mask0 = (data['OwnersAge'] <= higherBounds[0])  
mask1 = (data['OwnersAge'] <= higherBounds[1]) & (data['OwnersAge'] > higherBounds[0])
mask2 = (data['OwnersAge'] <= higherBounds[2]) & (data['OwnersAge'] > higherBounds[1])
mask3 = (data['OwnersAge'] <= higherBounds[3]) & (data['OwnersAge'] > higherBounds[2])
mask4 = (data['OwnersAge'] <= higherBounds[4]) & (data['OwnersAge'] > higherBounds[3])
mask5 = (data['OwnersAge'] <= higherBounds[5]) & (data['OwnersAge'] > higherBounds[4])
mask6 = (data['OwnersAge'] <= higherBounds[6]) & (data['OwnersAge'] > higherBounds[5])  
mask7 = (data['OwnersAge'] <= higherBounds[7]) & (data['OwnersAge'] > higherBounds[6])
mask8 = (data['OwnersAge'] > higherBounds[7])


maskk0 = (data['VehiculeAge'] <= high[0])
maskk1 = (data['VehiculeAge'] <= high[1]) & (data['VehiculeAge'] > high[0])
maskk2 = (data['VehiculeAge'] <= high[2]) & (data['VehiculeAge'] > high[1])
maskk3 = (data['VehiculeAge'] <= high[3]) & (data['VehiculeAge'] > high[2])
maskk4 = (data['VehiculeAge'] <= high[4]) & (data['VehiculeAge'] > high[3])
maskk5 = (data['VehiculeAge'] > high[4])

    
    
#using masks to assign groups
data.loc[mask0, 'OwnersAgeCat'] = 0
data.loc[mask1, 'OwnersAgeCat'] = 1
data.loc[mask2, 'OwnersAgeCat'] = 2
data.loc[mask3, 'OwnersAgeCat'] = 3
data.loc[mask4, 'OwnersAgeCat'] = 4
data.loc[mask5, 'OwnersAgeCat'] = 5
data.loc[mask6, 'OwnersAgeCat'] = 6
data.loc[mask7, 'OwnersAgeCat'] = 7
data.loc[mask8, 'OwnersAgeCat'] = 8

data.loc[maskk0, 'VehiculeAgeCat'] = 0
data.loc[maskk1, 'VehiculeAgeCat'] = 1
data.loc[maskk2, 'VehiculeAgeCat'] = 2
data.loc[maskk3, 'VehiculeAgeCat'] = 3
data.loc[maskk4, 'VehiculeAgeCat'] = 4    
data.loc[maskk5, 'VehiculeAgeCat'] = 5   

#now delete numeric variables 
data = data.drop(['OwnersAge', 'VehiculeAge'], axis = 1)
    
#convert all categorical to cat type 
catsNames= ['Gender', 'Zone', 'Class', 'BonusClass', 'OwnersAgeCat', 'VehiculeAgeCat']

for name in catsNames:
    data[name] = data[name].astype('category')


#now imitate the R preprocessing in order to have the same datasets. 
maskDurationZero = (data['Duration'] == 0)
data.loc[maskDurationZero, 'Duration'] = 0.00274

data = data.drop(data.index[3387])
data = data.drop(data.index[4198])
data = data.drop(data.index[15907])
data = data.drop(data.index[16075])


#now create dummy variables
data2 = pd.get_dummies(data, drop_first=True)


#save dataframe
data2.to_csv('C:/Users/kryst/Desktop/Poisson/Poisson-neural-network-insurance-pricing/catOnly.csv')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




