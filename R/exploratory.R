#This file's goal is to make an exploratory analysis of our dataset to get osme insights about what we are working with. 

path = "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/mcc.csv"

data = read.csv(path, sep = ",")

#Mapping K to F for clarity
library(plyr)
library(ggplot2)
library(gridExtra)
data$Gender = mapvalues(data$Gender, from = c("M", "K"), to = c("M", 'F'))

#Mapping zones 
#data$Zone = mapvalues(data$Zone, from = c("1", "2", "3", "4", "5", "6", "7"), to = c("Central Largest", "Suburbs", "Lesser Towns", "Small towns", "Northern towns", "Northern countryside", "Gotland"))

cc <- scales::seq_gradient_pal("black", "red", "Lab")(seq(0,1,length.out=80))
cc2 <- scales::seq_gradient_pal("black", "red", "Lab")(seq(0,1,length.out=20))

ggGender = ggplot(data=data, aes(x=Gender, fill = Gender)) + geom_bar(stat="count") +
  labs(title = "Gender repartition") + theme(plot.title = element_text(hjust = 0.5)) + 
  scale_fill_manual("legend", values = c("M" = "darkblue", "F" = "darkred", "3" = "black")) 
ggGender

ggOwnersAge = ggplot(data=data, aes(x=OwnersAge)) +
  geom_bar(stat="count", col = "black", aes(fill = ..count..)) + labs(title = "OwnersAge repartition") +
  theme(plot.title = element_text(hjust = 0.5)) + guides(fill=FALSE) + scale_fill_gradient("Count", low = "green", high = "red") 
ggOwnersAge

ggVehiculeAge = ggplot(data=data, aes(x=VehiculeAge)) + geom_bar(stat="count", col = "black", aes(fill = ..count..)) +
  labs(title = "VehiculeAge repartition") + theme(plot.title = element_text(hjust = 0.5)) +
  guides(fill=FALSE) + scale_fill_gradient("Count", low = "green", high = "red") 
ggVehiculeAge

ggZone = ggplot(data=data, aes(x=Zone)) + geom_bar(stat="count", col = "black", aes(fill = ..count..)) +
  labs(title = "Zone repartition") + theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient("Count", low = "green", high = "red") 
ggZone

ggClass = ggplot(data=data, aes(x=Class)) + geom_bar(stat="count", col = "black", aes(fill = ..count..)) +
  labs(title = "Class repartition") + theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient("Count", low = "green", high = "red") 
ggClass

ggBonus = ggplot(data=data, aes(x=BonusClass)) + geom_bar(stat="count", col = "black", aes(fill = ..count..)) +
  labs(title = "BonusClass repartition") + theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient("Count", low = "green", high = "red") 
ggBonus

ggClaims = ggplot(data=data, aes(x=NumberClaims)) + geom_bar(stat="count", col = "black", aes(fill = ..count..)) +
  labs(title = "Claims repartition") + theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient("Count", low = "green", high = "red") 
ggClaims

ggDuration = ggplot(data=data, aes(x=Duration)) + geom_histogram(breaks = seq(0,3,by = 0.1), col = "black", aes(fill = ..count..)) +
  labs(title = "Duration repartition") + theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient("Count", low = "green", high = "red") 
ggDuration

#do not take lines where cost is 0 or graph will be impossible to read because of the difference in counts. NOTE : some outliers arn't taken into the graph for a bette visualisation
ggCost = ggplot(data=data[data$ClaimCost >0,], aes(x = ClaimCost)) + geom_histogram(breaks = seq(0,200000, by = 500), aes(fill = ..count..)) +
  labs(title = "ClaimCost repartition") + theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient("Count", low = "green", high = "red") 
ggCost

#TODO : Plot for male/female claims numbers and claimcost ... Same with AGE ... Maybe two lines, M+F by age 




