#This is the GLM model of the project 

factors = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocData.csv", sep = ",")
y = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/NumberClaims.csv", sep = ',')
duration = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/Duration.csv", sep = ",")

data = cbind(factors, duration, y)

colnames(data)[length(data)-1] = "Duration"
colnames(data)[length(data)] = "NumberClaims"

library(caret)
library(caTools)
#install.packages("mlbench")
library(mlbench)
library(glmnet)
#using cross validation with K=10

#feature selection, using RFE https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/

control <- rfeControl(functions=rfFuncs, method="cv", number=10)

fit = glmnet(as.matrix(data[,1:26]), as.matrix(data[,27]), family = "poisson")
plot(fit, xvar = "dev", label = TRUE)

cvfit = cv.glmnet(as.matrix(data[,1:26]), as.matrix(data[,27]), family = "poisson")
plot(cvfit, xvar = "dev", label = TRUE)

#One line is simply the ?? lambda corresponding to the minimum MSE of the cross validation (your left dotted line) . 
#When adding one standard error to the minimum MSE value, you get a more regularized model, i.e. 
#one that performs favorable for predicting purposes. The ?? value belonging to it is denoted by the right dotted line. 

#accessing the values : 
cvfit$lambda.min; cvfit$lambda.1se #Ususally better to choose lambda1se because it reduces overfitting

coefLambda = coef(cvfit,s=cvfit$lambda.1se) #coefficient for lambda 1se

#Only 10 coefficients are kept from feature selection 

?barplot
barplot(coefLambda@x)


fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10
)

#Building model
glmt = caret::train(NumberClaims ~ . - Duration + offset(log(Duration)), data = data,
                    method = "glm", family = poisson(link = "log"), trControl = fitControl)

glmt
