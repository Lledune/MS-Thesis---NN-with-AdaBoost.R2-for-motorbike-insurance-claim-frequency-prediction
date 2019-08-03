#This is the GLM model of the project 

#dataTrain = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocFull.csv", sep = ",")
#oversampled
dataTrain = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocFull.csv", sep = ",")
duration = dataTrain[,22]
dataTrain = dataTrain[, -22]
nbc = dataTrain[,22]
dataTrain = dataTrain[, -22]

library(caret)
library(caTools)
#install.packages("mlbench")
library(mlbench)
library(glmnet)
#using cross validation with K=10

#Model simple
fit = glmnet(as.matrix(dataTrain[,1:21]), as.matrix(dataTrain[,22]), family = "poisson")
plot(fit, xvar = "dev", label = TRUE)

#Model with CV
cvfit = cv.glmnet(as.matrix(dataTrain[,1:21]), as.matrix(dataTrain[,22]), family = "poisson")
plot(cvfit, xvar = "dev", label = TRUE)

#One line is simply the ?? lambda corresponding to the minimum MSE of the cross validation (your left dotted line) . 
#When adding one standard error to the minimum MSE value, you get a more regularized model, i.e. 
#one that performs favorable for predicting purposes. The ?? value belonging to it is denoted by the right dotted line. 

#accessing the values : 
cvfit$lambda.min; cvfit$lambda.1se #Ususally better to choose lambda1se because it reduces overfitting

coefLambda = coef(cvfit,s=cvfit$lambda.min) #coefficient for lambda 1se
coefLambda
#Only 10 coefficients are kept from feature selection 
#TODO : Final model with selected features

dataTrain = as.matrix(dataTrain)
preds = predict(cvfit, newx = dataTrain[,1:21], s = cvfit$lambda.min)
predLambda = exp(preds)

dataTrain = as.data.frame(dataTrain)


#Deviance
devianceSingle = function(yt, yp, duration){
  
  if(yt == 0){
    return(2*duration*yp)
  }
  if(yt != 0){
    return(2*duration * (yt*log(yt) - yt*log(yp) - yt + yp))
  } 
}

devianceFull = function(yt, yp, duration){
  x = matrix(nrow = length(yt), ncol = 1)
  for(i in 1:length(x)){
    x[i] = devianceSingle(yt[i], yp[i], duration[i])
  }
  return(sum(x))
}

devianceFull(dataTrain$ClaimFrequency, predLambda, duration)

