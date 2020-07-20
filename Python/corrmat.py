import numpy as np 
import pandas as pd

import seaborn as sn
import matplotlib.pyplot as plt

#preprocessing 
data = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/mcc.csv")
data["ClaimFrequency"] = data["NumberClaims"]/data['Duration']
data["ClaimFrequency"] = data["ClaimFrequency"].fillna(0)

numsNames = ['OwnersAge', 'VehiculeAge', 'Duration', 'NumberClaims', 'ClaimFrequency', 'ClaimCost']
catsNames = ['Gender', 'Zone', 'Class', 'BonusClass']

for name in catsNames:
    data[name] = data[name].astype('category')

nums = data[numsNames]
cats = data.copy()
cats = cats.drop(numsNames, axis = 1)

dataDummies = pd.get_dummies(cats, drop_first=True)
data = pd.concat([nums, dataDummies], axis = 1)
#correlation matrix
plt.tight_layout()
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
corrmat = nums.corr()
plot = sn.heatmap(corrmat, cmap = plt.get_cmap('jet'), annot=True).set_title('Matrice de corrélation')
plt.show()

fig = plot.get_figure()
fig.savefig('C:/Users/kryst/Desktop/Poisson/Poisson-neural-network-insurance-pricing/Lyx/Images/corrmat.png')
