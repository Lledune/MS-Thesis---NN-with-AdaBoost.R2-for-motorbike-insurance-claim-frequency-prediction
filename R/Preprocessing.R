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

newData = cbind(dumData, data$OwnersAge, data$VehiculeAge, data$NumberClaims)
colnames(newData)[24:26] = c("OwnersAge", "VehiculeAge", "NumberClaims")

factors = newData[,1:25]
y = newData[,26]

write.csv(factors, row.names =  F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocData.csv")
write.csv(y, row.names = F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/NumberClaims.csv")
#removing duration and claimcost for now, we'll get them later 
duration = as.data.frame(data$Duration)
write.csv(duration, row.names = F, file = "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/Duration.csv")


#TODO : delete rows with duration = 0
sum(data$Duration[data$Duration == 0])
rowsDel = which(data$Duration == 0)
length(rowsDel)
#There is 2069 values where duration = 0
dataCheck = data[data$Duration == 0,]
table(dataCheck$NumberClaims)

#should we delete these ??
