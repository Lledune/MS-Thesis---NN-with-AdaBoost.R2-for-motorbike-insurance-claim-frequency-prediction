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

newData = cbind(dumData, data$OwnersAge, data$VehiculeAge, data$Duration, data$NumberClaims)
colnames(newData)[24:27] = c("OwnersAge", "VehiculeAge", "Duration", "NumberClaims")

#Normalize data 
normalize = function(vector){
  x = (vector - min(vector))/(max(vector) - min(vector))
  return(x)
}

newData = as.data.frame(newData)
newData$OwnersAge = normalize(newData$OwnersAge)
newData$VehiculeAge = normalize(newData$VehiculeAge)
newData$Duration = normalize(newData$Duration)
newData$NumberClaims = normalize(newData$NumberClaims)

#delete durations = 0 rows 
newData = newData[!(newData$Duration == 0), ]

#Test set building
smp_size <- floor(0.9 * nrow(newData))

## set the seed to make your partition reproducible
set.seed(100)
train_ind <- sample(seq_len(nrow(newData)), size = smp_size)

train <- newData[train_ind, ]
test <- newData[-train_ind, ]

#fullData
write.csv(newData, row.names =  F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocFull.csv")
write.csv(train, row.names =  F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocTrain.csv")
write.csv(test, row.names =  F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocTest.csv")

#should we delete these ??
newData = as.data.frame(newData)
indexes = !(durTrain == 0)
dataNoD = newData[indexes, ]
write.csv(dataNoD, row.names =  F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocNoD.csv")


