#This file will be used for the GLM building 
import numpy as np 
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import statsmodels.api as sm

#preprocessing 
data = pd.read_csv("c:/users/kryst/desktop/Poisson/Poisson-neural-network-insurance-pricing/preprocFull.csv")

y = data['NumberClaims']
d = data['Duration']
#dropping useless dimensions
data = data.drop(columns=["Duration", "NumberClaims", "ClaimFrequency"])
X = data
X = sm.add_constant(X) #adding constant bias 

model = sm.GLM(y, X, family = sm.families.Poisson(sm.families.links.log), exposure=d)
model_res = model.fit()
print(model_res.summary())

