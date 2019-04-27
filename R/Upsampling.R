#Upsampling of our data 

library(caret)

x = newData
x = as.data.frame(x)

x$NumberClaims = as.factor(x$NumberClaims)
samp = caret::upSample(x, x$NumberClaims)
samp$NumberClaims = as.character(samp$NumberClaims)
samp$NumberClaims = as.numeric(samp$NumberClaims)
samp = samp[,1:28]

table(samp$NumberClaims)

write.csv(samp, row.names =  F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/upsample.csv")

x = newData
x = as.data.frame(x)

x$NumberClaims = as.factor(x$NumberClaims)
samp = caret::downSample(x, x$NumberClaims)
samp$NumberClaims = as.character(samp$NumberClaims)
samp$NumberClaims = as.numeric(samp$NumberClaims)
samp = samp[,1:28]

table(samp$NumberClaims)

write.csv(samp, row.names =  F, "c:/users/lucien/desktop/Poisson-neural-network-insurance-pricing/downsample.csv")
