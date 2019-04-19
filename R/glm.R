#This is the GLM model of the project 

factorsTrain = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocDataTrain.csv", sep = ",")
yTrain = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/NumberClaimsTrain.csv", sep = ',')
durationTrain = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/DurationTrain.csv", sep = ",")
durationTrain = as.numeric(unlist(durationTrain))
dataTrain = cbind(factorsTrain, durationTrain, yTrain)



colnames(dataTrain)[length(dataTrain)-1] = "Duration"
colnames(dataTrain)[length(dataTrain)] = "NumberClaims"

library(caret)
library(caTools)
#install.packages("mlbench")
library(mlbench)
library(glmnet)
#using cross validation with K=10

#Model simple
fit = glmnet(as.matrix(dataTrain[,1:26]), as.matrix(dataTrain[,27]), family = "poisson")
plot(fit, xvar = "dev", label = TRUE)

#Model with CV
cvfit = cv.glmnet(as.matrix(dataTrain[,1:26]), as.matrix(dataTrain[,27]), family = "poisson")
plot(cvfit, xvar = "dev", label = TRUE)

#One line is simply the ?? lambda corresponding to the minimum MSE of the cross validation (your left dotted line) . 
#When adding one standard error to the minimum MSE value, you get a more regularized model, i.e. 
#one that performs favorable for predicting purposes. The ?? value belonging to it is denoted by the right dotted line. 

#accessing the values : 
cvfit$lambda.min; cvfit$lambda.1se #Ususally better to choose lambda1se because it reduces overfitting

coefLambda = coef(cvfit,s=cvfit$lambda.1se) #coefficient for lambda 1se
coefLambda
#Only 10 coefficients are kept from feature selection 
#TODO : Final model with selected features

dataTrain = as.matrix(dataTrain)
preds = predict(cvfit, newx = dataTrain[,1:26], s = cvfit$lambda.1se)
predLambda = exp(preds)

dataTrain = as.data.frame(dataTrain)

#Building model #TODO : Change offset +0001
glmt = glm(NumberClaims ~ . - Duration, offset=log(Duration+0.000001), data = dataTrain, family = poisson(link = "log"))

glmtCut = step(glmt)
preds = predict(glmtCut, dataTrain[,1:26])
predStep = exp(preds)

#Deviance
devianceSingle = function(yt, yp, duration){
  
  if(yt == 0){
    return(2*duration*yp)
  }
  if(yt != 0){
    return(2*duration * (yt*log(yt) - yt*log(yp) - yt + yp))
  } 
}
devianceSingle(dataTrain$NumberClaims[1], preds[1], durationTrain[1])

devianceFull = function(yt, yp, duration){
  x = matrix(nrow = length(yt), ncol = 1)
  for(i in 1:length(x)){
    x[i] = devianceSingle(yt[i], yp[i], duration[i])
  }
  return(x)
}

devFull = devianceFull(dataTrain$NumberClaims, preds, durationTrain)
devFull = round(devFull, 5)
devFull

#test models 
sum(devianceFull(dataTrain$NumberClaims, predStep, durationTrain)) #7321.062
sum(devianceFull(dataTrain$NumberClaims, predLambda, durationTrain)) #7211.471
