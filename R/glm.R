#This is the GLM model of the project 

factors = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/preprocData.csv", sep = ",")
y = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/NumberClaims.csv", sep = ',')
duration = read.csv("c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/Duration.csv", sep = ",")

data = cbind(factors, duration, y)

colnames(data)[length(data)-1] = "Duration"
colnames(data)[length(data)] = "NumberClaims"

library(caret)
library(caTools)

#using cross validation with K=10

fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10
)

#Building model
glmt = caret::train(NumberClaims ~ . - Duration + offset(log(Duration)), data = data,
                    method = "glm", family = poisson(link = "log"), trControl = fitControl)

glmt
