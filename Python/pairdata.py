#Code for pair plots

import numpy as np 
import pandas as pd

import seaborn as sn
import matplotlib.pyplot as plt

#preprocessing 
data = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/mcc.csv")
data["ClaimFrequency"] = data["NumberClaims"]/data['Duration']
data["ClaimFrequency"] = data["ClaimFrequency"].fillna(0)

numsNames = ['OwnersAge', 'VehiculeAge', 'Duration', 'ClaimCost']
catsNames = ['Gender', 'Zone', 'Class', 'BonusClass']

for name in catsNames:
    data[name] = data[name].astype('category')

nums = data[numsNames]
cats = data.copy()
cats = cats.drop(numsNames, axis = 1)

data2 = pd.get_dummies(data, columns=['Gender'])

#create boolean var for claims or not 
nums['Claim'] = data2['NumberClaims'] > 0
#create boolean for man or woman
nums['Man'] = data2['Gender_M'] == 1
nums['Zone'] = data2['Zone']
nums['Class'] = data2['Class']
nums['BonusClass'] = data2['BonusClass']

#mapping class number to string
ManDict = {0 : 'Woman',
           1 : 'Man'}
ZoneDict = {1 : 'Big City',
            2 : 'Medium City',
            3 : 'Small City',
            4 : 'Town',
            5 : 'North Sweden Cities',
            6 : 'North Sweden Campaign',
            7 : 'Gotland'
            }

for index, row in nums.iterrows():
    row['Man'] = ManDict[row['Man']]
    row['Zone'] = ZoneDict[row['Zone']]





plot1 = sn.pairplot(nums, hue = 'Claim', vars = numsNames, plot_kws={'alpha':0.5})

plot2 = sn.pairplot(nums, hue = 'Man', vars = numsNames, plot_kws={'alpha':0.5})

plot3 = sn.pairplot(nums, hue = 'Zone', vars = numsNames, plot_kws={'alpha':0.5})

plot4 = sn.pairplot(nums, hue = 'Class', vars = numsNames, plot_kws={'alpha':0.5})

plot5 = sn.pairplot(nums, hue = 'BonusClass', vars = numsNames, plot_kws={'alpha':0.5})

