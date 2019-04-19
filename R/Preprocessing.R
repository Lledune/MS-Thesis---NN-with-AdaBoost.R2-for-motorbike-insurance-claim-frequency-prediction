path = "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/mcc.csv"

data = read.csv(path, sep = ",")

#Mapping K to F for clarity
library(plyr)
library(ggplot2)
library(gridExtra)
data$Gender = mapvalues(data$Gender, from = c("M", "K"), to = c("M", 'F'))

#converting categoricals to factors in order to create dummies easeir 
data$Gender = as.factor(data$Gender)
data$Zone = as.factor(data$Zone)
data$BonusClass = as.factor(data$BonusClass)
data$Class = as.factor(data$Class)

#install.packages("caret")
#install.packages("caTools")
library(caret)
library(caTools)

dummies = dummyVars(NumberClaims ~ Gender + Zone + Class + BonusClass, data = data, drop2nd = T)  
dumData = predict(dummies, data)

newData = cbind(dumData, data$OwnersAge, data$VehiculeAge, data$NumberClaims, data$Duration)
colnames(newData)[24:27] = c("OwnersAge", "VehiculeAge", "NumberClaims", "Duration")

#Test set building

smp_size <- floor(0.9 * nrow(newData))

## set the seed to make your partition reproducible
set.seed(100)
train_ind <- sample(seq_len(nrow(newData)), size = smp_size)

train <- newData[train_ind, ]
test <- newData[-train_ind, ]


factorsTrain = train[,1:25]
factorsTest = test[,1:25]

yTrain = train[,26]
yTest = test[,26]

durTrain = train[,27]
durTest = test[,27]

#fullData
write.csv(newData, row.names =  F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocFull.csv")


#Factors
write.csv(factorsTrain, row.names =  F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocDataTrain.csv")
write.csv(factorsTest, row.names =  F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocDataTest.csv")

#y
write.csv(yTrain, row.names = F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/NumberClaimsTrain.csv")
write.csv(yTest, row.names = F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/NumberClaimsTest.csv")
#removing duration and claimcost for now, we'll get them later
duration = as.data.frame(data$Duration)
write.csv(durTrain, row.names = F, file = "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/DurationTrain.csv")
write.csv(durTest, row.names = F, file = "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/DurationTest.csv")

#TODO : delete rows with duration = 0
sum(data$Duration[data$Duration == 0])
rowsDel = which(data$Duration == 0)
length(rowsDel)
#There is 2069 values where duration = 0
dataCheck = data[data$Duration == 0,]
table(dataCheck$NumberClaims)

#should we delete these ??
