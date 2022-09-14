################################################################################
#***********************
################ ##############
##
##  install R 3.4.1 because of package dplyr or Use R studio
##
################################################################
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(version = "3.10")
BiocManager::install("galgo")
library(nnet)
library(ROCR)
library(caret)
library(doParallel) # parallel processing
library(dplyr) # Used by caret
library(pROC) # plot the ROC curve
library(randomForest)
library(deepnet)
library(neuralnet)
library(praznik)
library(e1071)
##***** Start *********####

##**** GSE34788 High Dimentional Heart Rythm Rate***#########

#*** Data Scaling ********* ####
#write.csv(GSE34788_all,"GSE34788_all.csv")
#GSE34788_all=read.csv("GSE34788_all.csv")
##******** TETFUND
GSE34788=read.csv("C:\\Users\\Kazeem\\Documents\\TETFUND KWASU\\Data_TETFUND\\GSE34788.csv")
dim(GSE34788)
GSE34788_HR=read.csv("C:\\Users\\Kazeem\\Documents\\TETFUND KWASU\\Data_TETFUND\\GSE34788_HR.csv")

GSE34788_all <- data.frame(GSE34788_HR,GSE34788[,-c(1)]) 
head(GSE34788_all[1:4])
## Data Scaling #######
GSE34788_ScaleX <- cbind(GSE34788_all[, 1], scale(GSE34788_all[,2:32960], center = T,scale = T))
dim(GSE34788_ScaleX )
GSE34788_dat_ScaleX <- as.data.frame(GSE34788_ScaleX)
head(GSE34788_dat_ScaleX[1:4])
dim(GSE34788_dat_ScaleX)
names(GSE34788_dat_ScaleX)[1] <- "response"
head(GSE34788_dat_ScaleX[,1:5])
write.csv(GSE34788_dat_ScaleX,"GSE34788_dat_ScaleX.csv")

## ***** Step One Filtering *****  ######
## ***** Using T-test at 99% Cutpoint **#####
GSE34788t.selection <- apply(GSE34788_dat_ScaleX[-1], 2, function(x)T.test=t.test(x ~ GSE34788_dat_ScaleX[,1], var.equal = F)$p.value)                                   
GSE34788t.selection <- sort(GSE34788t.selection[which(GSE34788t.selection <= 0.01)])
#write.csv(t.selection,file="t.selection_22.csv")
#t.selection=read.csv("t.selection_22.csv")
#GSE34788_dat=read.csv("GSE34788_dat.csv")
cat("\n", "Preliminary selection by t.statistic reporting p.values:", "\n"); utils::flush.console()
GSE34788dat_filtered <- subset(GSE34788_dat_ScaleX, select = c("response", names(GSE34788t.selection)))
print(length(GSE34788dat_filtered))
write.csv(GSE34788dat_filtered,"dat_filtered_GSE34788.csv")
library(plotly)
library(heatmaply)
head(dat_filtered[1:3])
dat_filtered=read.csv("dat_filtered_GSE34788.csv")
dim(dat_filtered)
dat_filtered_X=	dat_filtered[,-1]
heatmaply(dat_filtered_X[1:30], k_row = 2, k_col = 2)
heatmaply(cor(dat_filtered_X), k_row = 2, k_col = 2)


###****** #####
######### Process 2_1: Wrapper via  GA and Boruta ########################

#########****  GA Algorithm**** #############
dat_filteredGSE34788=read.csv("dat_filtered_GSE34788.csv")
dat_filteredGSE34788=dat_filteredGSE34788[,-1]
head(dat_filteredGSE34788[1:4])
trainXGSE34788=dat_filteredGSE34788[,-1]
trainyGSE34788=dat_filteredGSE34788[,1]
#trainy=ifelse(trainy==1,"Low Heart Rate","High Heart Rate")
yGSE34788=as.factor(trainyGSE34788)
dat_filtered2GSE34788=data.frame(yGSE34788,trainXGSE34788)
head(dat_filtered2GSE34788[1:4])

set.seed(1234)

registerDoParallel(4) # Registrer a parallel backend for train
getDoParWorkers() # check that there are 4 workers

ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)
## 
set.seed(10)
lev <- c(1,2)     # Set the levels

system.time(rf_gaGSE34788<- gafs(x = trainXGSE34788, y = yGSE34788,
                           iters = 100, # 100 generations of algorithm
                           popSize = 20, # population size for each generation
                           levels = lev,
                           gafsControl = ga_ctrl))

rf_gaGSE34788
plot(rf_gaGSE34788) # Plot mean fitness (AUC) by generation


final_GAGSE34788 <- rf_gaGSE34788$ga$final # Get features selected by GA
dat_GAGSE34788 <- subset(dat_filtered2GSE34788, select = c("yGSE34788", final_GAGSE34788))
head(dat_GAGSE34788[,1:4])
dim(dat_GAGSE34788)
write.csv(dat_GAGSE34788,"dat_GAGSE34788selected.csv")

#########****  Boruta Algorithm**** #############
library(Boruta)
dat_filteredGSE34788=read.csv("dat_filtered_GSE34788.csv")
dim(dat_filteredGSE34788)
head(dat_filteredGSE34788[1:4])
set.seed(1234)
system.time(boruta.train_GSE34788 <- Boruta(response~., data = dat_filteredGSE34788, doTrace = 2))
print(boruta.train_GSE34788)
## Selected Attributes
selectedFGSE34788=getSelectedAttributes(boruta.train_GSE34788, withTentative = FALSE)
boruta.dfGSE34788 <- attStats(boruta.train_GSE34788)
dat_BAGSE34788 <- subset(dat_filteredGSE34788, select = c("response",selectedFGSE34788))
write.csv(dat_BAGSE34788,"dat_BAGSE34788.csv")
IMPGSE34788=boruta.train_GSE34788$ImpHistory
IMPGSE34788_2=subset(IMPGSE34788,select=c(selectedFGSE34788))
write.csv(IMPGSE34788_2,"IMPGSE34788_2.csv")

IMPGSE34788_2=read.csv("IMPGSE34788_2.csv")
dim(IMPGSE34788_2)
head(IMPGSE34788_2[,1:4])
IMPGSE34788_M <- apply(IMPGSE34788_2, MARGIN = 2, FUN = median, na.rm = TRUE)
IMPGSE34788_M 
IMPGSE34788_Or <- order(IMPGSE34788_M, decreasing = FALSE)
library(RColorBrewer)
n <- 16
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

boxplot(IMPGSE34788_2[, IMPGSE34788_Or],col=sample(col_vector, n),
cex.axis = 1, las = 2,ylab=" Ranger Normalized Permutation Importance", 
boxwex = 0.6,main="GSE34788:Variable Importance")

## ***** 
dat_BAGSE34788=read.csv("dat_BAGSE34788.csv")
dim(dat_BAGSE34788)
head(dat_BAGSE34788[,1:4])
response=as.factor(dat_BAGSE34788$response)
Dat_BAGSE34788=data.frame(response,dat_BAGSE34788[,-c(1)])
head(Dat_BAGSE34788[,1:3])
levels(Dat_BAGSE34788$response)=make.names(levels(factor(Dat_BAGSE34788$response)))
selectedFGSE34788=colnames(Dat_BAGSE34788[,-1])

boruta.formulaGSE34788 <- formula(paste("response ~ ", 
                                paste(selectedFGSE34788, collapse = " + ")))
print(boruta.formulaGSE34788)
library(randomForest)
fitControl = trainControl(method = "repeatedcv",
                          classProbs = TRUE,
                          number = 10,
                          repeats = 5, 
                          index = createResample(Dat_BAGSE34788$response, 120),
                          summaryFunction = twoClassSummary,
                          verboseIter = FALSE)
rfBoruta.fitGSE34788 <- train(boruta.formulaGSE34788, 
                      data = Dat_BAGSE34788, 
                      trControl = fitControl,
                      tuneLength = 4,  # final value was mtry = 4
                      method = "rf",
                      metric = "ROC")
print(rfBoruta.fitGSE34788$finalModel)

#########################################################################
## Process 3: Embbeded ANN, DBN, ##
################ ########################################################

## Genetic Algorithm
dat_GAGSE34788selected=read.csv("dat_GAGSE34788selected.csv")
dim(dat_GAGSE34788selected)
head(dat_GAGSE34788selected[,1:3])
names(dat_GAGSE34788selected)[names(dat_GAGSE34788selected) == "yGSE34788"] <- "response"
head(dat_GAGSE34788selected[,1:3])

## ****** Monotone Multi-Layer Perceptron Neural Network ****
modelmonmlp<- train(as.factor(response)~., data=dat_GAGSE34788selected, trControl=train_control, method="monmlp")
modelmonmlp$results

## ****Deep Learning Net Learning *****************####
library(h2o)
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_GAGSE34788 <- h2o.importFile("dat_GAGSE34788selected.csv")
dim(dat_GAGSE34788selected)
y_GAGSE34788 = names(train_GAGSE34788)[1]
x_GAGSE34788 = names(train_GAGSE34788)[2:314]
train_GAGSE34788[,y_GAGSE34788] = as.factor(train_GAGSE34788[,y])
set.seed(1234)
model_GAGSE34788 = h2o.deeplearning(x=x_GAGSE34788, 
                         y=y_GAGSE34788, 
                         training_frame=train_GAGSE34788, 
                         validation_frame=train_GAGSE34788, 
                         distribution = "multinomial",
                         activation = "RectifierWithDropout",
                         hidden = c(10,10,10,10),
                         input_dropout_ratio = 0.2,
                         l1 = 1e-5,
                         epochs = 50)

print(model_GAGSE34788)

plot(h2o.performance(model_GAGSE34788,valid=T),type='roc',col="blue",lwd=2)
text(0.2,0.7,"AUC=0.9944")

# **** Fit DNN ***** ####
library(deepnet)
dim(dat_GAGSE34788selected)
set.seed(2345)
xx_GAGSE34788=dat_GAGSE34788selected[,-1]
yy_GAGSE34788=dat_GAGSE34788selected[,1]
nn_GAGSE34788 <- nn.train(as.matrix(xx_GAGSE34788), yy_GAGSE34788, hidden = c(5))
yyy_GAGSE34788 = nn.predict(nn_GAGSE34788, xx_GAGSE34788)
print(head(yyy_GAGSE34788))
yhat = matrix(0,length(yyy_GAGSE34788),1)
yhat[which(yyy_GAGSE34788 > mean(yyy_GAGSE34788))] = 2
yhat[which(yyy_GAGSE34788<= mean(yyy_GAGSE34788))] = 1
cm_GAGSE34788 = table(yy_GAGSE34788,yhat)
print(cm_GAGSE34788)
print(sum(diag(cm_GAGSE34788))/sum(cm_GAGSE34788))

## Boruta Algorithm
dat_BAGSE34788=read.csv("dat_BAGSE34788.csv")
dim(dat_BAGSE34788)
head(dat_BAGSE34788[,1:3])
names(dat_BAGSE34788)
## VIP
library(RSNNS)
library(NeuralNetTools)
require(clusterGeneration)
library(nnet)
x_BAGSE34788 <- dat_BAGSE34788[, c(
 "X8091764" ,"X8078382", "X8108424",
"X8159243", "X7952739", "X8140752", "X7897172",
"X8031646", "X7968254", "X8023195", "X7959023",
"X8027381", "X7917148" ,"X7956152", "X7901982",
"X8016628"
)]
y_BAGSE34788 <- dat_BAGSE34788[, "response"]
model_mlp_BAGSE34788<- mlp(x_BAGSE34788, y_BAGSE34788, size = 16,linout=T)
plot.nnet(model_mlp_BAGSE34788)
#import 'gar.fun' from beckmw's Github - this is Garson's algorithm
mod_BAGSE34788<-nnet(x_BAGSE34788,y_BAGSE34788,size=16,linout=T)
source_gist('6206737')
#use the function on the model created aboveBAGSE34788
gar.fun('y_BAGSE34788',mod_BAGSE34788)
## Variable Important
rel.imp_BAGSE34788<-gar.fun('y_BAGSE34788',mod_BAGSE34788,bar.plot=F)$rel.imp
rel.imp_BAGSE34788=sort(cbind(rel.imp_BAGSE34788))
write.csv(rel.imp_BAGSE34788,"rel.imp_BAGSE34788.csv")
#color vector based on relative importance of input values
cols<-colorRampPalette(c('green','red'))(num.vars)[rank(rel.imp)]

##
#plotting function
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#plot model with new color vector
#separate colors for input vectors using a list for 'circle.col'
plot(mod_BAGSE34788,circle.col=list(cols,'lightblue'))



## ****** Monotone Multi-Layer Perceptron Neural Network ****
set.seed(3452)
modelmonmlp<- train(as.factor(response)~., data=dat_BAGSE34788, trControl=train_control, method="monmlp")
modelmonmlp$results

## ****Deep Learning Net Learning *****************####
library(h2o)
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_BAGSE34788 <- h2o.importFile("dat_BAGSE34788.csv")
dim(dat_BAGSE34788)
y_BAGSE34788 = names(train_BAGSE34788)[1]
x_BAGSE34788 = names(train_BAGSE34788)[2:16]
train_BAGSE34788[,y_BAGSE34788] = as.factor(train_BAGSE34788[,y_BAGSE34788])
set.seed(1234)
model_BAGSE34788 = h2o.deeplearning(x=x_BAGSE34788, 
                                    y=y_BAGSE34788, 
                                    training_frame=train_BAGSE34788, 
                                    validation_frame=train_BAGSE34788, 
                                    distribution = "multinomial",
                                    activation = "RectifierWithDropout",
                                    hidden = c(10,10,10,10),
                                    input_dropout_ratio = 0.2,
                                    l1 = 1e-5,
                                    epochs = 50)

print(model_BAGSE34788)

plot(h2o.performance(model_BAGSE34788,valid=T),type='roc',col="blue",lwd=2)
text(0.2,0.7,"AUC=0.9427")

# **** Fit DNN ***** ####
library(deepnet)
dim(dat_BAGSE34788)
set.seed(2345)
xx_BAGSE34788=dat_BAGSE34788[,-1]
yy_BAGSE34788=dat_BAGSE34788[,1]
nn_BAGSE34788 <- nn.train(as.matrix(xx_BAGSE34788), yy_BAGSE34788, hidden = c(5))
yyy_BAGSE34788 = nn.predict(nn_BAGSE34788, xx_BAGSE34788)
print(head(yyy_BAGSE34788))
yhat = matrix(0,length(yyy_BAGSE34788),1)
yhat[which(yyy_BAGSE34788 > mean(yyy_BAGSE34788))] = 2
yhat[which(yyy_BAGSE34788<= mean(yyy_BAGSE34788))] = 1
cm_BAGSE34788 = table(yy_BAGSE34788,yhat)
print(cm_BAGSE34788)
print(sum(diag(cm_BAGSE34788))/sum(cm_BAGSE34788))

#### ************ Data Two ****** ######
################# GSE86569 ############################
#########################################################
GSE86569=read.csv("C:\\Users\\Kazeem\\Documents\\TETFUND KWASU\\Data_TETFUND\\GSE86569.csv")
dim(GSE86569)
head(GSE86569[,1:4])
## Data Scaling #######
GSE86569_ScaleX <- cbind(GSE86569[, 1], scale(GSE86569[,-1], center = T,scale = T))
dim(GSE86569_ScaleX )
GSE86569_dat <- as.data.frame(GSE86569_ScaleX)
head(GSE86569_dat [,1:4])
names(GSE86569_dat)[1] <- "response"
head(GSE86569_dat[1:5])
write.csv(GSE86569_dat,"GSE86569_dat.csv")

## ***** Step One Filtering *****  ######
## ***** Using T-test at 99% Cutpoint **#####
GSE86569t.selection<- apply(GSE86569_dat[-1], 2, function(x)T.test=t.test(x ~ GSE86569_dat[,1], var.equal = F)$p.value)                                   
t.selection_GSE86569 <- sort(GSE86569t.selection[which(GSE86569t.selection <= 0.01)])
write.csv(t.selection_GSE86569 ,file="t.selection_GSE86569 .csv")
#t.selection=read.csv("t.selection_GSE62727.csv")
#GSE62727_dat=read.csv("GSE34788_dat.csv")
cat("\n", "Preliminary selection by t.statistic reporting p.values:", "\n"); utils::flush.console()
dat_filteredGSE86569 <- subset(GSE86569_dat, select = c("response", names(t.selection_GSE86569)))
print(length(dat_filteredGSE86569 ))
write.csv(dat_filteredGSE86569,"dat_filteredGSE86569.csv")
library(plotly)
library(heatmaply)
head(dat_filteredGSE86569[1:3])
#dat_filteredGSE86569=read.csv("dat_filteredGSE86569.csv")
head(dat_filteredGSE86569[,1:4])
dim(dat_filteredGSE86569)
dat_filtered_XGSE86569=dat_filteredGSE86569[,-1]
par(mfrow=c(2,1))
heatmaply(dat_filtered_XGSE86569[1:30], k_row = 2, k_col = 2,
main = "Heatmap of the GSE86569 micro-array data" )
heatmaply(cor(dat_filtered_XGSE86569[1:30]), k_row = 2, 
k_col = 2,na.rm = T, main = "Correlation matrix heatmap for GSE86569 micro-array data")

###****** #####
######### Process 2_1: Wrapper via  GA and Boruta ########################

#########****  GA Algorithm**** #############
dat_filteredGSE86569=read.csv("dat_filteredGSE86569.csv")
dim(dat_filteredGSE86569)
head(dat_filteredGSE86569[1:4])
trainXGSE86569=dat_filteredGSE86569[,-1]
trainyGSE86569=dat_filteredGSE86569[,1]
#trainy=ifelse(trainy==1,"Low Heart Rate","High Heart Rate")
yGSE86569=as.factor(trainyGSE86569)
dat_filtered2GSE86569=data.frame(yGSE86569,trainXGSE86569)
head(dat_filtered2GSE86569[1:4])

set.seed(1234)
registerDoParallel(4) # Registrer a parallel backend for train
getDoParWorkers() # check that there are 4 workers

ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)
## 
set.seed(10)
lev <- c(1,2)     # Set the levels

system.time(rf_gaGSE86569<- gafs(x = trainXGSE86569, y = yGSE86569,
                                 iters = 100, # 100 generations of algorithm
                                 popSize = 20, # population size for each generation
                                 levels = lev,
                                 gafsControl = ga_ctrl))

rf_gaGSE86569
plot(rf_gaGSE86569) # Plot mean fitness (AUC) by generation

final_GAGSE86569 <- rf_gaGSE86569$ga$final # Get features selected by GA
dat_GAGSE86569 <- subset(dat_filtered2GSE86569, select = c("yGSE86569", final_GAGSE86569))
head(dat_GAGSE86569[,1:4])
dim(dat_GAGSE86569)
write.csv(dat_GAGSE86569,"dat_GAGSE86569selected.csv")

#########****  Boruta Algorithm**** #############
library(Boruta)
dat_filteredGSE86569=read.csv("dat_filteredGSE86569.csv")
dim(dat_filteredGSE86569)
head(dat_filteredGSE86569[1:4])
set.seed(1234)
system.time(boruta.train_GSE86569 <- Boruta(response~., data = dat_filteredGSE86569, doTrace = 2))
print(boruta.train_GSE86569)
## Selected Attributes
selectedFGSE86569=getSelectedAttributes(boruta.train_GSE86569, withTentative = FALSE)
boruta.dfGSE86569 <- attStats(boruta.train_GSE86569)
dat_BAGSE86569 <- subset(dat_filteredGSE86569, select = c("response",selectedFGSE86569))
dim(dat_BAGSE86569)
write.csv(dat_BAGSE86569,"dat_BAGSE86569.csv")
IMPGSE86569=boruta.train_GSE86569$ImpHistory
IMPGSE86569_2=subset(IMPGSE86569,select=c(selectedFGSE86569))
write.csv(IMPGSE86569_2,"IMPGSE86569_2.csv")

#IMPGSE86569_2=read.csv("IMPGSE86569_2.csv")
dim(IMPGSE86569_2)
head(IMPGSE86569_2[,1:4])
IMPGSE86569_M <- apply(IMPGSE86569_2, MARGIN = 2, FUN = median, na.rm = TRUE)
IMPGSE86569_M 
IMPGSE86569_Or <- order(IMPGSE86569_M, decreasing = FALSE)
library(RColorBrewer)
n <- 12
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

boxplot(IMPGSE86569_2[, IMPGSE86569_Or],col=sample(col_vector, n),
        cex.axis = 1, las = 2,ylab=" Ranger Normalized Permutation Importance", 
        boxwex = 0.6,main="GSE86569:Variable Importance")

## ***** 
dat_BAGSE86569=read.csv("dat_BAGSE86569.csv")
dim(dat_BAGSE86569)
head(dat_BAGSE86569[,1:4])
response=as.factor(dat_BAGSE86569$response)
Dat_BAGSE86569=data.frame(response,dat_BAGSE86569[,-c(1)])
head(Dat_BAGSE86569[,1:3])
levels(Dat_BAGSE86569$response)=make.names(levels(factor(Dat_BAGSE86569$response)))
selectedFGSE86569=colnames(Dat_BAGSE86569[,-1])

boruta.formulaGSE86569 <- formula(paste("response ~ ", 
                                        paste(selectedFGSE86569, collapse = " + ")))
print(boruta.formulaGSE86569)
library(randomForest)
fitControl = trainControl(method = "repeatedcv",
                          classProbs = TRUE,
                          number = 10,
                          repeats = 5, 
                          index = createResample(Dat_BAGSE86569$response, 22),
                          summaryFunction = twoClassSummary,
                          verboseIter = FALSE)
rfBoruta.fitGSE86569 <- train(boruta.formulaGSE86569, 
                              data = Dat_BAGSE86569, 
                              trControl = fitControl,
                              tuneLength = 4,  # final value was mtry = 4
                              method = "rf",
                              metric = "ROC")
print(rfBoruta.fitGSE86569$finalModel)



#########################################################################
## Process 2: Embbeded ANN, DBN, ##
################ ########################################################

# Genetic Algorithm
dat_GAGSE86569selected=read.csv("dat_GAGSE86569selected.csv")
dim(dat_GAGSE86569selected)
head(dat_GAGSE86569selected[,1:3])
names(dat_GAGSE86569selected)[names(dat_GAGSE86569selected) == "yGSE86569"] <- "response"
head(dat_GAGSE86569selected[,1:3])

## ****** Monotone Multi-Layer Perceptron Neural Network ****
train_control <- trainControl(method="cv", number=10)
set.seed(23456)
modelmonmlp<- train(as.factor(response)~., data=dat_GAGSE86569selected, trControl=train_control, method="monmlp")
modelmonmlp$results

## ****Deep Learning Net Learning *****************####
library(h2o)
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_GAGSE86569 <- h2o.importFile("dat_GAGSE86569selected.csv")
dim(train_GAGSE86569)
head(train_GAGSE86569[,1:3])
y_GAGSE86569 = names(train_GAGSE86569)[1]
x_GAGSE86569 = names(train_GAGSE86569)[2:26]
train_GAGSE86569[,y_GAGSE86569] = as.factor(train_GAGSE86569[,y_GAGSE86569])
set.seed(1234)
model_GAGSE86569 = h2o.deeplearning(x=x_GAGSE86569, 
                                    y=y_GAGSE86569, 
                                    training_frame=train_GAGSE86569, 
                                    validation_frame=train_GAGSE86569, 
                                    distribution = "multinomial",
                                    activation = "RectifierWithDropout",
                                    hidden = c(10,10,10,10),
                                    input_dropout_ratio = 0.2,
                                    l1 = 1e-5,
                                    epochs = 50)

print(model_GAGSE86569)

plot(h2o.performance(model_GAGSE86569,valid=T),type='roc',col="blue",lwd=2)
text(0.2,0.7,"AUC=1.0000")

# **** Fit DNN ***** ####
library(deepnet)
dim(dat_GAGSE86569selected)
set.seed(2345)
xx_GAGSE86569=dat_GAGSE86569selected[,-1]
yy_GAGSE86569=dat_GAGSE86569selected[,1]
nn_GAGSE86569 <- nn.train(as.matrix(xx_GAGSE86569), yy_GAGSE86569, hidden = c(5))
yyy_GAGSE86569 = nn.predict(nn_GAGSE86569, xx_GAGSE86569)
print(head(yyy_GAGSE86569))
yhat = matrix(0,length(yyy_GAGSE86569),1)
yhat[which(yyy_GAGSE86569 > mean(yyy_GAGSE86569))] = 2
yhat[which(yyy_GAGSE86569<= mean(yyy_GAGSE86569))] = 1
cm_GAGSE86569 = table(yy_GAGSE86569,yhat)
print(cm_GAGSE86569)
print(sum(diag(cm_GAGSE86569))/sum(cm_GAGSE86569))

## Boruta Algorithm
dat_BAGSE86569=read.csv("dat_BAGSE86569.csv")
dim(dat_BAGSE86569)
head(dat_BAGSE86569[,1:3])
names(dat_BAGSE86569)
## VIP
library(RSNNS)
library(NeuralNetTools)
require(clusterGeneration)
library(nnet)
x_BAGSE86569 <- dat_BAGSE86569[, c(
"TC03000507.hg.1", "TC03001702.hg.1",
"TC03001945.hg.1", "TC03000224.hg.1", "TC01003707.hg.1",
"TC02002084.hg.1", "TC01001151.hg.1", "TC02000620.hg.1",
"TC01004718.hg.1", "X3145076_st","TC02004344.hg.1",
"TC01005931.hg.1"
)]
y_BAGSE86569 <- dat_BAGSE86569[, "response"]
model_mlp_BAGSE86569<- mlp(x_BAGSE86569, y_BAGSE86569, size = 12,linout=T)
plot.nnet(model_mlp_BAGSE86569)
#import 'gar.fun' from beckmw's Github - this is Garson's algorithm
mod_BAGSE86569<-nnet(x_BAGSE86569,y_BAGSE86569,size=12,linout=T)
source_gist('6206737')
#use the function on the model created aboveBAGSE34788
gar.fun('y_BAGSE86569',mod_BAGSE86569)
## Variable Important
rel.imp_BAGSE86569<-gar.fun('y_BAGSE86569',mod_BAGSE86569,bar.plot=F)$rel.imp
rel.imp_BAGSE86569=sort(cbind(rel.imp_BAGSE86569))
write.csv(rel.imp_BAGSE86569,"rel.imp_BAGSE86569.csv")
#color vector based on relative importance of input values
cols<-colorRampPalette(c('green','red'))(num.vars)[rank(rel.imp)]

##
#plotting function
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#plot model with new color vector
#separate colors for input vectors using a list for 'circle.col'
plot(mod_BAGSE86569,circle.col=list(cols,'lightblue'))


## ****** Monotone Multi-Layer Perceptron Neural Network ****
set.seed(3452)
modelmonmlp<- train(as.factor(response)~., data=dat_BAGSE86569, trControl=train_control, method="monmlp")
modelmonmlp$results

## ****Deep Learning Net Learning *****************####
library(h2o)
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_BAGSE86569 <- h2o.importFile("dat_BAGSE86569.csv")
dim(dat_BAGSE86569)
y_BAGSE86569 = names(train_BAGSE86569)[1]
x_BAGSE86569 = names(train_BAGSE86569)[2:12]
train_BAGSE86569[,y_BAGSE86569] = as.factor(train_BAGSE86569[,y_BAGSE86569])
head(train_BAGSE86569[,1:4])
set.seed(1234)
model_BAGSE86569 = h2o.deeplearning(x=x_BAGSE86569, 
                                    y=y_BAGSE86569, 
                                    training_frame=train_BAGSE86569, 
                                    validation_frame=train_BAGSE86569, 
                                    distribution = "multinomial",
                                    activation = "RectifierWithDropout",
                                    hidden = c(10,10,10,10),
                                    input_dropout_ratio = 0.2,
                                    l1 = 1e-5,
                                    epochs = 50)

print(model_BAGSE86569)

plot(h2o.performance(model_BAGSE86569,valid=T),type='roc',col="blue",lwd=2)
text(0.2,0.7,"AUC=0.9833")

# **** Fit DNN ***** ####
library(deepnet)
dim(dat_BAGSE86569)
set.seed(2345)
xx_BAGSE86569=dat_BAGSE86569[,-1]
yy_BAGSE86569=dat_BAGSE86569[,1]
nn_BAGSE86569 <- nn.train(as.matrix(xx_BAGSE86569), yy_BAGSE86569, hidden = c(5))
yyy_BAGSE86569 = nn.predict(nn_BAGSE86569, xx_BAGSE86569)
print(head(yyy_BAGSE86569))
yhat = matrix(0,length(yyy_BAGSE86569),1)
yhat[which(yyy_BAGSE86569 > mean(yyy_BAGSE86569))] = 2
yhat[which(yyy_BAGSE86569<= mean(yyy_BAGSE86569))] = 1
cm_BAGSE86569 = table(yy_BAGSE86569,yhat)
print(cm_BAGSE86569)
print(sum(diag(cm_BAGSE86569))/sum(cm_BAGSE86569))

################# Data THREE ############################
#########################################################
GSE115574=read.csv("C:\\Users\\Kazeem\\Documents\\TETFUND KWASU\\Data_TETFUND\\GSE115574.csv")
dim(GSE115574)
head(GSE115574[,1:4])
## Data Scaling #######
GSE115574_ScaleX <- cbind(GSE115574[, 1], scale(GSE115574[,-1], center = T,scale = T))
dim(GSE115574_ScaleX )
GSE115574_dat <- as.data.frame(GSE115574_ScaleX)
head(GSE115574_dat [,1:4])
names(GSE115574_dat)[1] <- "response"
head(GSE115574_dat[1:5])
write.csv(GSE115574_dat,"GSE115574_dat.csv")

## ***** Step One Filtering *****  ######
## ***** Using T-test at 99% Cutpoint **#####
GSE115574t.selection<- apply(GSE115574_dat[-1], 2, function(x)T.test=t.test(x ~ GSE115574_dat[,1], var.equal = F)$p.value)                                   
t.selection_GSE115574 <- sort(GSE115574t.selection[which(GSE115574t.selection <= 0.01)])
write.csv(t.selection_GSE115574 ,file="t.selection_GSE115574 .csv")
#t.selection=read.csv("t.selection_GSE62727.csv")
#GSE62727_dat=read.csv("GSE34788_dat.csv")
dat_filteredGSE115574 <- subset(GSE115574_dat, select = c("response", names(t.selection_GSE115574)))
print(length(dat_filteredGSE115574 ))
write.csv(dat_filteredGSE115574,"dat_filteredGSE115574.csv")
library(plotly)
library(heatmaply)
head(dat_filteredGSE115574[1:3])
#dat_filteredGSE115574=read.csv("dat_filteredGSE115574.csv")
head(dat_filteredGSE115574[,1:4])
dim(dat_filteredGSE115574)
dat_filtered_XGSE115574=dat_filteredGSE115574[,-1]
heatmaply(dat_filtered_XGSE115574[1:30], k_row = 2, 
k_col = 2, main = "Heatmap of the GSE115574 micro-array data")
heatmaply(cor(dat_filtered_XGSE115574[1:30]), k_row = 2, 
k_col = 2,na.rm = T,main = "Correlation matrix heatmap for GSE115574 micro-array data")

###****** #####
######### Process 2_1: Wrapper via  GA and Boruta ########################

#########****  GA Algorithm**** #############
dat_filteredGSE115574=read.csv("dat_filteredGSE115574.csv")
dim(dat_filteredGSE115574)
head(dat_filteredGSE115574[1:4])
trainXGSE115574=dat_filteredGSE115574[,-1]
trainyGSE115574=dat_filteredGSE115574[,1]
#trainy=ifelse(trainy==1,"Low Heart Rate","High Heart Rate")
yGSE115574=as.factor(trainyGSE115574)
dat_filtered2GSE115574=data.frame(yGSE115574,trainXGSE115574)
head(dat_filtered2GSE115574[1:4])

set.seed(1234)
registerDoParallel(4) # Registrer a parallel backend for train
getDoParWorkers() # check that there are 4 workers

ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)
## 
set.seed(10)
lev <- c(1,2)     # Set the levelsGSE115574

system.time(rf_gaGSE115574<- gafs(x = trainXGSE115574, y = yGSE115574,
                                 iters = 100, # 100 generations of algorithm
                                 popSize = 20, # population size for each generation
                                 levels = lev,
                                 gafsControl = ga_ctrl))

rf_gaGSE115574
plot(rf_gaGSE115574) # Plot mean fitness (AUC) by generation

final_GAGSE115574 <- rf_gaGSE115574$ga$final # Get features selected by GA
dat_GAGSE115574 <- subset(dat_filtered2GSE115574, select = c("yGSE115574", final_GAGSE115574))
head(dat_GAGSE115574[,1:4])
dim(dat_GAGSE115574)
write.csv(dat_GAGSE115574,"dat_GAGSE115574selected.csv")

#########****  Boruta Algorithm**** #############
library(Boruta)
dat_filteredGSE115574=read.csv("dat_filteredGSE115574.csv")
dim(dat_filteredGSE115574)
head(dat_filteredGSE115574[1:4])
set.seed(1234)
system.time(boruta.train_GSE115574 <- Boruta(response~., data = dat_filteredGSE115574, doTrace = 2))
print(boruta.train_GSE115574)
## Selected Attributes
selectedFGSE115574=getSelectedAttributes(boruta.train_GSE115574, withTentative = FALSE)
boruta.dfGSE115574 <- attStats(boruta.train_GSE115574)
dat_BAGSE115574 <- subset(dat_filteredGSE115574, select = c("response",selectedFGSE115574))
dim(dat_BAGSE115574)
write.csv(dat_BAGSE115574,"dat_BAGSE115574.csv")
IMPGSE115574=boruta.train_GSE115574$ImpHistory
IMPGSE115574_2=subset(IMPGSE115574,select=c(selectedFGSE115574))
write.csv(IMPGSE115574_2,"IMPGSE115574.csv")

#IMPGSE115574_2=read.csv("IMPGSE115574_2.csv")
dim(IMPGSE115574_2)
head(IMPGSE115574_2[,1:4])
IMPGSE115574_M <- apply(IMPGSE115574_2, MARGIN = 2, FUN = median, na.rm = TRUE)
IMPGSE115574_M 
IMPGSE115574_Or <- order(IMPGSE115574_M, decreasing = FALSE)
library(RColorBrewer)
n <- 19
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

boxplot(IMPGSE115574_2[, IMPGSE115574_Or],col=sample(col_vector, n),
        cex.axis = 0.7, las = 2,ylab=" Ranger Normalized Permutation Importance", 
        boxwex = 0.6,main="GSE115574:Variable Importance")

## ***** 
dat_BAGSE115574=read.csv("dat_BAGSE115574.csv")
dim(dat_BAGSE115574)
head(dat_BAGSE115574[,1:4])
response=as.factor(dat_BAGSE115574$response)
Dat_BAGSE115574=data.frame(response,dat_BAGSE115574[,-c(1)])
head(Dat_BAGSE115574[,1:3])
levels(Dat_BAGSE115574$response)=make.names(levels(factor(Dat_BAGSE115574$response)))
selectedFGSE115574=colnames(Dat_BAGSE115574[,-1])

boruta.formulaGSE115574 <- formula(paste("response ~ ", 
                                        paste(selectedFGSE115574, collapse = " + ")))
print(boruta.formulaGSE115574)
library(randomForest)
fitControl = trainControl(method = "repeatedcv",
                          classProbs = TRUE,
                          number = 10,
                          repeats = 5, 
                          index = createResample(Dat_BAGSE115574$response, 59),
                          summaryFunction = twoClassSummary,
                          verboseIter = FALSE)
rfBoruta.fitGSE115574 <- train(boruta.formulaGSE115574, 
                              data = Dat_BAGSE115574, 
                              trControl = fitControl,
                              tuneLength = 4,  # final value was mtry = 4
                              method = "rf",
                              metric = "ROC")
print(rfBoruta.fitGSE115574$finalModel)

#########################################################################
## Process 2: Embbeded ANN, DBN, ##
################ ########################################################

# Genetic Algorithm
dat_GAGSE115574selected=read.csv("dat_GAGSE115574selected.csv")
dim(dat_GAGSE115574selected)
head(dat_GAGSE115574selected[,1:3])
names(dat_GAGSE115574selected)[names(dat_GAGSE115574selected) == "yGSE115574"] <- "response"
head(dat_GAGSE115574selected[,1:3])

## ****** Monotone Multi-Layer Perceptron Neural Network ****
train_control <- trainControl(method="cv", number=10)
set.seed(23456)
modelmonmlpGSE115574<- train(as.factor(response)~., data=dat_GAGSE115574selected, trControl=train_control, method="monmlp")
modelmonmlpGSE115574$results

## ****Deep Learning Net Learning *****************####
library(h2o)
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_GAGSE115574 <- h2o.importFile("dat_GAGSE115574selected.csv")
dim(train_GAGSE115574)
head(train_GAGSE115574[,1:3])
y_GAGSE115574 = names(train_GAGSE115574)[1]
x_GAGSE115574 = names(train_GAGSE115574)[2:59]
train_GAGSE115574[,y_GAGSE115574] = as.factor(train_GAGSE115574[,y_GAGSE115574])
set.seed(1234)
model_GAGSE115574 = h2o.deeplearning(x=x_GAGSE115574, 
                                    y=y_GAGSE115574, 
                                    training_frame=train_GAGSE115574, 
                                    validation_frame=train_GAGSE115574, 
                                    distribution = "multinomial",
                                    activation = "RectifierWithDropout",
                                    hidden = c(10,10,10,10),
                                    input_dropout_ratio = 0.2,
                                    l1 = 1e-5,
                                    epochs = 50)

print(model_GAGSE115574)

plot(h2o.performance(model_GAGSE115574,valid=T),type='roc',col="red",lwd=2)
text(0.2,0.7,"AUC=0.983871")

# **** Fit DNN ***** ####
library(deepnet)
dim(dat_GAGSE115574selected)
set.seed(2345)
xx_GAGSE115574=dat_GAGSE115574selected[,-1]
yy_GAGSE115574=dat_GAGSE115574selected[,1]
nn_GAGSE115574 <- nn.train(as.matrix(xx_GAGSE115574), yy_GAGSE115574, hidden = c(5))
yyy_GAGSE115574 = nn.predict(nn_GAGSE115574, xx_GAGSE115574)
print(head(yyy_GAGSE115574))
yhat = matrix(0,length(yyy_GAGSE115574),1)
yhat[which(yyy_GAGSE115574 > mean(yyy_GAGSE115574))] = 2
yhat[which(yyy_GAGSE115574<= mean(yyy_GAGSE115574))] = 1
cm_GAGSE115574 = table(yy_GAGSE115574,yhat)
print(cm_GAGSE115574)
print(sum(diag(cm_GAGSE115574))/sum(cm_GAGSE115574))

## Boruta Algorithm
dat_BAGSE115574=read.csv("dat_BAGSE115574.csv")
dim(dat_BAGSE115574)
head(dat_BAGSE115574[,1:3])
names(dat_BAGSE115574)
## VIP
library(RSNNS)
library(NeuralNetTools)
require(clusterGeneration)
library(nnet)
x_BAGSE115574 <- dat_BAGSE115574[, c(
"X204260_at",    "X1555869_a_at",
"X201522_x_at",  "X1553746_a_at", "X201995_at",   
"X205952_at" ,   "X203458_at",    "X1560164_at",  
"X1559399_s_at", "X202196_s_at",  "X200839_s_at", 
"X1552522_at",   "X1554018_at",   "X201869_s_at", 
"X203333_at" ,   "X201030_x_at",  "X204647_at",   
"X201141_at" ,   "X1558587_at" 
)]
y_BAGSE115574 <- dat_BAGSE115574[, "response"]
model_mlp_BAGSE115574<- mlp(x_BAGSE115574, y_BAGSE115574, size = 19,linout=T)
plot.nnet(model_mlp_BAGSE115574)
#import 'gar.fun' from beckmw's Github - this is Garson's algorithm
mod_BAGSE115574<-nnet(x_BAGSE115574,y_BAGSE115574,size=19,linout=T)
source_gist('6206737')
#use the function on the model created aboveBAGSE34788
gar.fun('y_BAGSE115574',mod_BAGSE115574)
## Variable Important
rel.imp_BAGSE115574<-gar.fun('y_BAGSE115574',mod_BAGSE115574,bar.plot=F)$rel.imp
rel.imp_BAGSE115574=sort(cbind(rel.imp_BAGSE115574))
write.csv(rel.imp_BAGSE115574,"rel.imp_BAGSE115574.csv")
#color vector based on relative importance of input values
cols<-colorRampPalette(c('green','red'))(num.vars)[rank(rel.imp)]

##
#plotting function
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#plot model with new color vector
#separate colors for input vectors using a list for 'circle.col'
plot(mod_BAGSE115574,circle.col=list(cols,'lightblue'))






## ****** Monotone Multi-Layer Perceptron Neural Network ****
set.seed(3452)
modelmonmlpGSE115574<- train(as.factor(response)~., data=dat_BAGSE115574, trControl=train_control, method="monmlp")
modelmonmlpGSE115574$results

## ****Deep Learning Net Learning *****************####
library(h2o)
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_BAGSE115574 <- h2o.importFile("dat_BAGSE115574.csv")
dim(dat_BAGSE115574)
y_BAGSE115574 = names(train_BAGSE115574)[1]
x_BAGSE115574 = names(train_BAGSE115574)[2:19]
train_BAGSE115574[,y_BAGSE115574] = as.factor(train_BAGSE115574[,y_BAGSE115574])
head(train_BAGSE115574[,1:4])
set.seed(1234)
model_BAGSE115574 = h2o.deeplearning(x=x_BAGSE115574, 
                                    y=y_BAGSE115574, 
                                    training_frame=train_BAGSE115574, 
                                    validation_frame=train_BAGSE115574, 
                                    distribution = "multinomial",
                                    activation = "RectifierWithDropout",
                                    hidden = c(10,10,10,10),
                                    input_dropout_ratio = 0.2,
                                    l1 = 1e-5,
                                    epochs = 50)

print(model_BAGSE115574)

plot(h2o.performance(model_BAGSE115574,valid=T),type='roc',col="blue",lwd=2)
text(0.2,0.7,"AUC=0.9804")

# **** Fit DNN ***** ####
library(deepnet)
dim(dat_BAGSE115574)
set.seed(2345)
xx_BAGSE115574=dat_BAGSE115574[,-1]
yy_BAGSE115574=dat_BAGSE115574[,1]
nn_BAGSE115574 <- nn.train(as.matrix(xx_BAGSE115574), yy_BAGSE115574, hidden = c(5))
yyy_BAGSE115574 = nn.predict(nn_BAGSE115574, xx_BAGSE115574)
print(head(yyy_BAGSE115574))
yhat = matrix(0,length(yyy_BAGSE115574),1)
yhat[which(yyy_BAGSE115574 > mean(yyy_BAGSE115574))] = 2
yhat[which(yyy_BAGSE115574<= mean(yyy_BAGSE115574))] = 1
cm_BAGSE115574 = table(yy_BAGSE115574,yhat)
print(cm_BAGSE115574)
print(sum(diag(cm_BAGSE115574))/sum(cm_BAGSE115574))


################# Data FOUR ############################
########################################################
GSE112266=read.csv("C:\\Users\\Kazeem\\Documents\\TETFUND KWASU\\Data_TETFUND\\GSE112266.csv")
dim(GSE112266)
head(GSE112266[,1:4])

## Data Scaling #######
GSE112266_ScaleX <- cbind(GSE112266[, 1], scale(GSE112266[,-1], center = T,scale = T))
dim(GSE112266_ScaleX )
GSE112266_dat <- as.data.frame(GSE112266_ScaleX)
head(GSE112266_dat [,1:4])
names(GSE112266_dat)[1] <- "response"
head(GSE112266_dat[1:5])
write.csv(GSE112266_dat,"GSE112266_dat.csv")

## ***** Step One Filtering *****  ######
## ***** Using T-test at 99% Cutpoint **#####
GSE112266t.selection<- apply(GSE112266_dat[-1], 2, function(x)T.test=t.test(x ~ GSE112266_dat[,1], var.equal = F)$p.value)                                   
t.selection_GSE112266 <- sort(GSE112266t.selection[which(GSE112266t.selection <= 0.01)])
write.csv(t.selection_GSE112266 ,file="t.selection_GSE112266 .csv")
#t.selection=read.csv("t.selection_GSE62727.csv")
#GSE62727_dat=read.csv("GSE34788_dat.csv")
cat("\n", "Preliminary selection by t.statistic reporting p.values:", "\n"); utils::flush.console()
dat_filteredGSE112266 <- subset(GSE112266_dat, select = c("response", names(t.selection_GSE112266)))
print(length(dat_filteredGSE112266))
write.csv(dat_filteredGSE112266,"dat_filteredGSE112266_Current.csv")
library(plotly)
library(heatmaply)
head(dat_filteredGSE112266_Current[1:3])
dat_filteredGSE112266=read.csv("dat_filteredGSE112266_Current.csv")
head(dat_filteredGSE112266[,1:4])
dim(dat_filteredGSE112266)
dat_filtered_XGSE112266=dat_filteredGSE112266[,-c(1,2)]
head(dat_filtered_XGSE112266[,1:3])
heatmaply(dat_filtered_XGSE112266[1:30], k_row = 2, 
k_col = 2, main = "Heatmap of the GSE112266 micro-array data")
heatmaply(cor(dat_filtered_XGSE112266[1:30]), k_row = 2, 
k_col = 2,na.rm = T, main = "Correlation matrix heatmap for GSE1122664 micro-array data")

###****** #####
######### Process 2_1: Wrapper via  GA and Boruta ########################

#########****  GA Algorithm**** #############
dat_filteredGSE112266=read.csv("dat_filteredGSE112266_Current.csv")
dim(dat_filteredGSE112266)
head(dat_filteredGSE112266[,1:3])
trainXGSE112266=dat_filteredGSE112266[,-1]
trainyGSE112266=dat_filteredGSE112266[,1]
yGSE112266=as.factor(trainyGSE112266)
dat_filtered2GSE112266=data.frame(yGSE112266,trainXGSE112266)
head(dat_filtered2GSE112266[1:4])

set.seed(1234)
registerDoParallel(4) # Registrer a parallel backend for train
getDoParWorkers() # check that there are 4 workers

ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)
## 
set.seed(10)
lev <- c(1,2)     # Set the levelsGSE115574

system.time(rf_gaGSE112266<- gafs(x = trainXGSE112266, y = yGSE112266,
                                  iters = 100, # 100 generations of algorithm
                                  popSize = 20, # population size for each generation
                                  levels = lev,
                                  gafsControl = ga_ctrl))

rf_gaGSE112266
plot(rf_gaGSE112266) # Plot mean fitness (AUC) by generation

final_GAGSE112266 <- rf_gaGSE112266$ga$final # Get features selected by GA
dat_GAGSE112266 <- subset(dat_filtered2GSE112266, select = c("yGSE112266", final_GAGSE112266))
head(dat_GAGSE112266[,1:4])
dim(dat_GAGSE112266)
write.csv(dat_GAGSE112266,"dat_GAGSE112266selected.csv")

#########****  Boruta Algorithm**** #############
library(Boruta)
dat_filteredGSE112266=read.csv("dat_filteredGSE112266_Current.csv")
dim(dat_filteredGSE112266)
head(dat_filteredGSE112266[,1:3])
set.seed(1234)
system.time(boruta.train_GSE112266 <- Boruta(response~., data = dat_filteredGSE112266, doTrace = 2))
print(boruta.train_GSE112266)
## Selected Attributes
selectedFGSE112266=getSelectedAttributes(boruta.train_GSE112266, withTentative = FALSE)
boruta.dfGSE112266 <- attStats(boruta.train_GSE112266)
dat_BAGSE112266 <- subset(dat_filteredGSE112266, select = c("response",selectedFGSE112266))
dim(dat_BAGSE112266)
write.csv(dat_BAGSE112266,"dat_BAGSE112266.csv")
IMPGSE112266=boruta.train_GSE112266$ImpHistory
IMPGSE112266_2=subset(IMPGSE112266,select=c(selectedFGSE112266))
write.csv(IMPGSE112266_2,"IMPGSE112266.csv")

#IMPGSE115574_2=read.csv("IMPGSE115574_2.csv")
dim(IMPGSE112266_2)
head(IMPGSE112266_2[,1:4])
IMPGSE112266_M <- apply(IMPGSE112266_2, MARGIN = 2, FUN = median, na.rm = TRUE)
IMPGSE112266_M 
IMPGSE112266_Or <- order(IMPGSE112266_M, decreasing = FALSE)
library(RColorBrewer)
n <- 17
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

boxplot(IMPGSE112266_2[, IMPGSE112266_Or],col=sample(col_vector, n),
        cex.axis = 1, las = 2,ylab=" Ranger Normalized Permutation Importance", 
        boxwex = 0.6,main="GSE112266:Variable Importance")

## ***** 
dat_BAGSE115574=read.csv("dat_BAGSE112266.csv")
dim(dat_BAGSE112266)
head(dat_BAGSE112266[,1:4])
response=as.factor(dat_BAGSE112266$response)
Dat_BAGSE112266=data.frame(response,dat_BAGSE112266[,-c(1)])
head(Dat_BAGSE112266[,1:3])
levels(Dat_BAGSE112266$response)=make.names(levels(factor(Dat_BAGSE112266$response)))
selectedFGSE112266=colnames(Dat_BAGSE112266[,-1])

boruta.formulaGSE112266 <- formula(paste("response ~ ", 
                                         paste(selectedFGSE112266, collapse = " + ")))
print(boruta.formulaGSE112266)
library(randomForest)
fitControl = trainControl(method = "repeatedcv",
                          classProbs = TRUE,
                          number = 10,
                          repeats = 5, 
                          index = createResample(Dat_BAGSE112266$response, 53),
                          summaryFunction = twoClassSummary,
                          verboseIter = FALSE)
rfBoruta.fitGSE112266 <- train(boruta.formulaGSE112266, 
                               data = Dat_BAGSE112266, 
                               trControl = fitControl,
                               tuneLength = 6,  # final value was mtry = 4
                               method = "rf",
                               metric = "ROC")
print(rfBoruta.fitGSE112266$finalModel)

#########################################################################
## Process 2: Embbeded ANN, DBN, ##
################ ########################################################

# Genetic Algorithm
dat_GAGSE112266selected=read.csv("dat_GAGSE112266selected.csv")
dim(dat_GAGSE112266selected)
head(dat_GAGSE112266selected[,1:3])
names(dat_GAGSE112266selected)[names(dat_GAGSE112266selected) == "yGSE112266"] <- "response"
head(dat_GAGSE112266selected[,1:3])

## ****** Monotone Multi-Layer Perceptron Neural Network ****
train_control <- trainControl(method="cv", number=10)
set.seed(23456)
modelmonmlpGSE112266<- train(as.factor(response)~., data=dat_GAGSE112266selected, trControl=train_control, method="monmlp")
modelmonmlpGSE112266$results

## ****Deep Learning Net Learning *****************####
library(h2o)
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_GAGSE112266 <- h2o.importFile("dat_GAGSE112266selected.csv")
dim(train_GAGSE112266)
head(train_GAGSE112266[,1:3])
y_GAGSE112266 = names(train_GAGSE112266)[1]
x_GAGSE112266 = names(train_GAGSE112266)[2:20]
train_GAGSE112266[,y_GAGSE112266] = as.factor(train_GAGSE112266[,y_GAGSE112266])
set.seed(1234)
model_GAGSE112266 = h2o.deeplearning(x=x_GAGSE112266, 
                                     y=y_GAGSE112266, 
                                     training_frame=train_GAGSE112266, 
                                     validation_frame=train_GAGSE112266, 
                                     distribution = "multinomial",
                                     activation = "RectifierWithDropout",
                                     hidden = c(10,10,10,10),
                                     input_dropout_ratio = 0.2,
                                     l1 = 1e-5,
                                     epochs = 50)

print(model_GAGSE112266)

plot(h2o.performance(model_GAGSE112266,valid=T),type='roc',col="red",lwd=2)
text(0.2,0.55,"AUC=0.9088")

# **** Fit DNN ***** ####
library(deepnet)
dim(dat_GAGSE112266selected)
set.seed(2345)
xx_GAGSE112266=dat_GAGSE112266selected[,-1]
yy_GAGSE112266=dat_GAGSE112266selected[,1]
nn_GAGSE112266 <- nn.train(as.matrix(xx_GAGSE112266), yy_GAGSE112266, hidden = c(5))
yyy_GAGSE112266 = nn.predict(nn_GAGSE112266, xx_GAGSE112266)
print(head(yyy_GAGSE112266))
yhat = matrix(0,length(yyy_GAGSE112266),1)
yhat[which(yyy_GAGSE112266 > mean(yyy_GAGSE112266))] = 2
yhat[which(yyy_GAGSE112266<= mean(yyy_GAGSE112266))] = 1
cm_GAGSE112266 = table(yy_GAGSE112266,yhat)
print(cm_GAGSE112266)
print(sum(diag(cm_GAGSE112266))/sum(cm_GAGSE112266))

## Boruta Algorithm
dat_BAGSE112266=read.csv("dat_BAGSE112266.csv")
dim(dat_BAGSE112266)
head(dat_BAGSE112266[,1:3])
names(dat_BAGSE112266)
## VIP
library(RSNNS)
library(NeuralNetTools)
require(clusterGeneration)
library(nnet)
x_BAGSE112266 <- dat_BAGSE112266[, c(
"E2F_Dp2b", "PCNA",          
"cdc20" , "Vde" ,"HSF4.6a",       
"Sigma70.6" ,"HSF4.3a","Sigma70.4",     
"Myb5R","Myb3R2",         "petA",          
"Myb1R_SHAQKYF4", "HSF4.2c",  "psaD",          
"Myb1R10" ,       "Cpd3" ,    "BUB1.MAD3" 
)]
y_BAGSE112266 <- dat_BAGSE112266[, "response"]
model_mlp_BAGSE112266<- mlp(x_BAGSE112266, y_BAGSE112266, size = 19,linout=T)
plot.nnet(model_mlp_BAGSE112266)
#import 'gar.fun' from beckmw's Github - this is Garson's algorithm
mod_BAGSE112266<-nnet(x_BAGSE112266,y_BAGSE112266,size=19,linout=T)
source_gist('6206737')
#use the function on the model created aboveBAGSE34788
gar.fun('y_BAGSE112266',mod_BAGSE112266)
## Variable Important
rel.imp_BAGSE112266<-gar.fun('y_BAGSE112266',mod_BAGSE112266,bar.plot=F)$rel.imp
rel.imp_BAGSE112266=sort(cbind(rel.imp_BAGSE112266))
write.csv(rel.imp_BAGSE112266,"rel.imp_BAGSE112266.csv")
#color vector based on relative importance of input values
cols<-colorRampPalette(c('green','red'))(num.vars)[rank(rel.imp)]

##
#plotting function
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#plot model with new color vector
#separate colors for input vectors using a list for 'circle.col'
plot(mod_BAGSE112266,circle.col=list(cols,'lightblue'))





## ****** Monotone Multi-Layer Perceptron Neural Network ****
set.seed(3452)
modelmonmlpGSE112266<- train(as.factor(response)~., data=dat_BAGSE112266, trControl=train_control, method="monmlp")
modelmonmlpGSE112266$results

## ****Deep Learning Net Learning *****************####
library(h2o)
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_BAGSE112266 <- h2o.importFile("dat_BAGSE112266.csv")
dim(dat_BAGSE112266)
y_BAGSE112266 = names(train_BAGSE112266)[1]
x_BAGSE112266 = names(train_BAGSE112266)[2:18]
train_BAGSE112266[,y_BAGSE112266] = as.factor(train_BAGSE112266[,y_BAGSE112266])
head(train_BAGSE112266[,1:4])
set.seed(1234)
model_BAGSE112266 = h2o.deeplearning(x=x_BAGSE112266, 
                                     y=y_BAGSE112266, 
                                     training_frame=train_BAGSE112266, 
                                     validation_frame=train_BAGSE112266, 
                                     distribution = "multinomial",
                                     activation = "RectifierWithDropout",
                                     hidden = c(10,10,10,10),
                                     input_dropout_ratio = 0.2,
                                     l1 = 1e-5,
                                     epochs = 50)

print(model_BAGSE112266)

plot(h2o.performance(model_BAGSE112266,valid=T),type='roc',col="blue",lwd=2)
text(0.25,0.65,"BAGSE112266 AUC=0.9084")

# **** Fit DNN ***** ####
library(deepnet)
dim(dat_BAGSE112266)
set.seed(2345)
xx_BAGSE112266=dat_BAGSE112266[,-1]
yy_BAGSE112266=dat_BAGSE112266[,1]
nn_BAGSE112266 <- nn.train(as.matrix(xx_BAGSE112266), yy_BAGSE112266, hidden = c(5))
yyy_BAGSE112266 = nn.predict(nn_BAGSE112266, xx_BAGSE112266)
print(head(yyy_BAGSE112266))
yhat = matrix(0,length(yyy_BAGSE112266),1)
yhat[which(yyy_BAGSE112266 > mean(yyy_BAGSE112266))] = 2
yhat[which(yyy_BAGSE112266<= mean(yyy_BAGSE112266))] = 1
cm_BAGSE112266 = table(yy_BAGSE112266,yhat)
print(cm_BAGSE112266)
print(sum(diag(cm_BAGSE112266))/sum(cm_BAGSE112266))


################# Data FIVE ############################
########################################################
library(Biobase)
library(GEOquery)

# load series and platform data from GEO

gset <- getGEO("GSE68475", GSEMatrix =TRUE, getGPL=FALSE)
if (length(gset) > 1) idx <- grep("GPL15018", attr(gset, "names")) else idx <- 1
gset <- gset[[idx]]
write.csv(gset,"GSE68475.csv")
GSE68475=read.csv("C:\\Users\\Kazeem\\Documents\\TETFUND KWASU\\Data_TETFUND\\GSE68475.csv")
dim(GSE68475)
head(GSE68475[,1:4])


## Data Scaling #######
GSE68475_ScaleX <- cbind(GSE68475[, 1], scale(GSE68475[,-1], center = T,scale = T))
dim(GSE68475_ScaleX )
GSE68475_dat <- as.data.frame(GSE68475_ScaleX)
head(GSE68475_dat [,1:4])
names(GSE68475_dat)[1] <- "response"
head(GSE68475_dat[1:5])
write.csv(GSE68475_dat,"GSE68475_dat.csv")

## ***** Step One Filtering *****  ######
## ***** Using T-test at 99% Cutpoint **#####
GSE68475t.selection<- apply(GSE68475_dat[-1], 2, function(x)T.test=t.test(x ~ GSE68475_dat[,1], var.equal = F)$p.value)                                   
t.selection_GSE68475 <- sort(GSE68475t.selection[which(GSE68475t.selection <= 0.01)])
write.csv(t.selection_GSE68475 ,file="t.selection_GSE68475 .csv")
#t.selection=read.csv("t.selection_GSE62727.csv")
#GSE62727_dat=read.csv("GSE34788_dat.csv")
cat("\n", "Preliminary selection by t.statistic reporting p.values:", "\n"); utils::flush.console()
dat_filteredGSE68475 <- subset(GSE68475_dat, select = c("response", names(t.selection_GSE68475)))
print(length(dat_filteredGSE68475))
write.csv(dat_filteredGSE68475,"dat_filteredGSE68475.csv")
library(plotly)
library(heatmaply)
head(dat_filteredGSE68475[1:3])
dat_filteredGSE68475=read.csv("dat_filteredGSE68475.csv")
head(dat_filteredGSE68475[,1:4])
dim(dat_filteredGSE68475)
dat_filtered_XGSE68475=dat_filteredGSE68475[,-c(1)]
head(dat_filtered_XGSE68475[,1:3])
heatmaply(dat_filtered_XGSE68475[1:30], k_row = 2, 
k_col = 2, main = "Heatmap of the GSE68475 micro-array data")
heatmaply(cor(dat_filtered_XGSE68475[1:30]), k_row = 2, 
k_col = 2,na.rm = T, main = "Correlation matrix heatmap for GSE68475 micro-array data")

###****** #####
######### Process 2_1: Wrapper via  GA and Boruta ########################

#########****  GA Algorithm**** #############
dat_filteredGSE68475=read.csv("dat_filteredGSE68475.csv")
dim(dat_filteredGSE68475)
head(dat_filteredGSE68475[,1:3])
trainXGSE68475=dat_filteredGSE68475[,-1]
trainyGSE68475=dat_filteredGSE68475[,1]
yGSE68475=as.factor(trainyGSE68475)
dat_filtered2GSE68475=data.frame(yGSE68475,trainXGSE68475)
head(dat_filtered2GSE68475[1:4])

set.seed(1234)
registerDoParallel(4) # Registrer a parallel backend for train
getDoParWorkers() # check that there are 4 workers

ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)
## 
set.seed(10)
lev <- c(1,2)     # Set the levelsGSE115574

system.time(rf_gaGSE68475<- gafs(x = trainXGSE68475, y = yGSE68475,
                                  iters = 100, # 100 generations of algorithm
                                  popSize = 20, # population size for each generation
                                  levels = lev,
                                  gafsControl = ga_ctrl))

rf_gaGSE68475
plot(rf_gaGSE68475) # Plot mean fitness (AUC) by generation

final_GAGSE68475 <- rf_gaGSE68475$ga$final # Get features selected by GA
dat_GAGSE68475 <- subset(dat_filtered2GSE68475, select = c("yGSE68475", final_GAGSE68475))
head(dat_GAGSE68475[,1:4])
dim(dat_GAGSE68475)
write.csv(dat_GAGSE68475,"dat_GAGSE68475selected.csv")

#########****  Boruta Algorithm**** #############
library(Boruta)
dat_filteredGSE68475=read.csv("dat_filteredGSE68475.csv")
dim(dat_filteredGSE68475)
head(dat_filteredGSE68475[,1:3])
set.seed(1234)
system.time(boruta.train_GSE68475 <- Boruta(response~., data = dat_filteredGSE68475, doTrace = 0))
print(boruta.train_GSE68475)
## Selected Attributes
selectedFGSE68475=getSelectedAttributes(boruta.train_GSE68475, withTentative = FALSE)
boruta.dfGSE68475 <- attStats(boruta.train_GSE68475)
dat_BAGSE68475 <- subset(dat_filteredGSE68475, select = c("response",selectedFGSE68475))
dim(dat_BAGSE68475)
write.csv(dat_BAGSE68475,"dat_BAGSE68475.csv")
IMPGSE68475=boruta.train_GSE68475$ImpHistory
IMPGSE68475_2=subset(IMPGSE68475,select=c(selectedFGSE68475))
write.csv(IMPGSE68475_2,"IMPGSE68475.csv")

IMPGSE68475_2=read.csv("IMPGSE68475.csv")
dim(IMPGSE68475_2)
head(IMPGSE68475_2[,1:4])
IMPGSE68475_M <- apply(IMPGSE68475_2, MARGIN = 2, FUN = median, na.rm = TRUE)
IMPGSE68475_M 
IMPGSE68475_Or <- order(IMPGSE68475_M, decreasing = FALSE)
library(RColorBrewer)
n <- 23
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

boxplot(IMPGSE68475_2[, IMPGSE68475_Or],col=sample(col_vector, n),
        cex.axis = 1, las = 2,ylab=" Ranger Normalized Permutation Importance", 
        boxwex = 0.6,main="GSE68475:Variable Importance")

## ***** 
dat_BAGSE68475=read.csv("dat_BAGSE68475.csv")
dim(dat_BAGSE68475)
head(dat_BAGSE68475[,1:4])
response=as.factor(dat_BAGSE68475$response)
Dat_BAGSE68475=data.frame(response,dat_BAGSE68475[,-c(1)])
head(Dat_BAGSE68475[,1:3])
levels(Dat_BAGSE68475$response)=make.names(levels(factor(Dat_BAGSE68475$response)))
selectedFGSE68475=colnames(Dat_BAGSE68475[,-1])

boruta.formulaGSE68475 <- formula(paste("response ~ ", 
                                         paste(selectedFGSE68475, collapse = " + ")))
print(boruta.formulaGSE68475)
library(randomForest)
fitControl = trainControl(method = "repeatedcv",
                          classProbs = TRUE,
                          number = 10,
                          repeats = 5, 
                          index = createResample(Dat_BAGSE68475$response,21),
                          summaryFunction = twoClassSummary,
                          verboseIter = FALSE)
rfBoruta.fitGSE68475 <- train(boruta.formulaGSE68475, 
                               data = Dat_BAGSE68475, 
                               trControl = fitControl,
                               tuneLength = 6,  # final value was mtry = 4
                               method = "rf",
                               metric = "ROC")
print(rfBoruta.fitGSE68475$finalModel)

#########################################################################
## Process 2: Embbeded ANN, DBN, ##
################ ########################################################

# Genetic Algorithm
dat_GAGSE68475selected=read.csv("dat_GAGSE68475selected.csv")
dim(dat_GAGSE68475selected)
head(dat_GAGSE68475selected[,1:3])
names(dat_GAGSE68475selected)[names(dat_GAGSE68475selected) == "yGSE68475"] <- "response"
head(dat_GAGSE68475selected[,1:3])

## ****** Monotone Multi-Layer Perceptron Neural Network ****
train_control <- trainControl(method="cv", number=10)
set.seed(23456)
modelmonmlpGSE68475<- train(as.factor(response)~., data=dat_GAGSE68475selected, trControl=train_control, method="monmlp")
modelmonmlpGSE68475$results

## ****Deep Learning Net Learning *****************####
library(h2o)
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_GAGSE68475 <- h2o.importFile("dat_GAGSE68475selected.csv")
dim(train_GAGSE68475)
head(train_GAGSE68475[,1:3])
y_GAGSE68475 = names(train_GAGSE68475)[1]
x_GAGSE68475 = names(train_GAGSE68475)[2:18]
train_GAGSE68475[,y_GAGSE68475] = as.factor(train_GAGSE68475[,y_GAGSE68475])
set.seed(1234)
model_GAGSE68475 = h2o.deeplearning(x=x_GAGSE68475, 
                                     y=y_GAGSE68475, 
                                     training_frame=train_GAGSE68475, 
                                     validation_frame=train_GAGSE68475, 
                                     distribution = "multinomial",
                                     activation = "RectifierWithDropout",
                                     hidden = c(10,10,10,10),
                                     input_dropout_ratio = 0.2,
                                     l1 = 1e-5,
                                     epochs = 50)

print(model_GAGSE68475)

plot(h2o.performance(model_GAGSE68475,valid=T),type='roc',col="red",lwd=2)
text(0.2,0.55,"GAGSE68475 AUC=0.9727")

# **** Fit DNN ***** ####
library(deepnet)
dim(dat_GAGSE68475selected)
set.seed(2345)
xx_GAGSE68475=dat_GAGSE68475selected[,-1]
yy_GAGSE68475=dat_GAGSE68475selected[,1]
nn_GAGSE68475 <- nn.train(as.matrix(xx_GAGSE68475), yy_GAGSE68475, hidden = c(5))
yyy_GAGSE68475 = nn.predict(nn_GAGSE68475, xx_GAGSE68475)
print(head(yyy_GAGSE68475))
yhat = matrix(0,length(yyy_GAGSE68475),1)
yhat[which(yyy_GAGSE68475 > mean(yyy_GAGSE68475))] = 2
yhat[which(yyy_GAGSE68475<= mean(yyy_GAGSE68475))] = 1
cm_GAGSE68475 = table(yy_GAGSE68475,yhat)
print(cm_GAGSE68475)
print(sum(diag(cm_GAGSE68475))/sum(cm_GAGSE68475))


dat_GAGSE68475=read.csv("dat_GAGSE68475selected.csv")
dim(dat_GAGSE68475)
head(dat_GAGSE68475[,1:3])
## VIP
library(RSNNS)
library(NeuralNetTools)
require(clusterGeneration)
library(nnet)
x_GAGSE68475 <- dat_GAGSE68475[, c(
"X17706",	"X17451",	"X17154",	"X16390",	"X4492",	"X1560",	"X17412",
"X13186",	"X17203",	"X16619",	"X16616",	"X3506",	"X9363",	
"X12761",	"X8645",	"X5345",	"X11372"
)]
y_GAGSE68475 <- dat_GAGSE68475[, "yGSE68475"]
model_mlp_GAGSE68475<- mlp(x_GAGSE68475, y_GAGSE68475, size = 17,linout=T)
plot.nnet(model_mlp_GAGSE68475)
#import 'gar.fun' from beckmw's Github - this is Garson's algorithm
mod_GAGSE68475<-nnet(x_GAGSE68475,y_GAGSE68475,size=17,linout=T)
source_gist('6206737')
#use the function on the model created above
gar.fun('y_GAGSE68475',mod_GAGSE68475)
## Variable Important
rel.imp_GAGSE68475<-gar.fun('y_GAGSE68475',mod_GAGSE68475,bar.plot=F)$rel.imp
rel.imp_GAGSE68475=sort(cbind(rel.imp_GAGSE68475))
write.csv(rel.imp_GAGSE68475,"rel.imp_GAGSE68475.csv")
mm=read.csv("rel.imp_GAGSE68475.csv")
mmm=sort(mm[,2])
#color vector based on relative importance of input values
cols<-colorRampPalette(c('green','red'))(num.vars)[rank(rel.imp)]

##
#plotting function
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#plot model with new color vector
#separate colors for input vectors using a list for 'circle.col'
plot(mod_GAGSE68475,circle.col=list(cols,'lightblue'))



## Boruta Algorithm
dat_BAGSE68475=read.csv("dat_BAGSE68475.csv")
dim(dat_BAGSE68475)
head(dat_BAGSE68475[,1:3])

## ****** Monotone Multi-Layer Perceptron Neural Network ****
set.seed(3452)
modelmonmlpGSE68475<- train(as.factor(response)~., data=dat_BAGSE68475, trControl=train_control, method="monmlp")
modelmonmlpGSE68475$results

library(RSNNS)
library(NeuralNetTools)
require(clusterGeneration)
library(nnet)
x_BAGSE68475 <- dat_BAGSE68475[, c("X18297","X17706",	"X2527",	"X7369",	"X14009",	"X10200",
"X4151",	"X7045",	"X11004",	"X12346",	"X11410",	"X11999",	"X15614",	"X16361",
"X15815",	"X5132",	"X16478",	"X10604",	"X10655",	"X344",	"X1560",	"X3075",	"X3830"
)]
y_BAGSE68475 <- dat_BAGSE68475[, "response"]
model_mlp_BAGSE68475<- mlp(x_BAGSE68475, y_BAGSE68475, size = 23,linout=T)
plot.nnet(model_mlp_BAGSE68475)
#import 'gar.fun' from beckmw's Github - this is Garson's algorithm
mod_BAGSE68475<-nnet(x_BAGSE68475,y_BAGSE68475,size=23,linout=T)
source_gist('6206737')
#use the function on the model created above
gar.fun('y_BAGSE68475',mod_BAGSE68475)

## Variable Important
rel.imp_BAGSE68475<-gar.fun('y_BAGSE68475',mod_BAGSE68475,bar.plot=F)$rel.imp
rel.imp_BAGSE68475=sort(cbind(rel.imp_BAGSE68475))
write.csv(rel.imp_BAGSE68475,"rel.imp_BAGSE68475.csv")
#color vector based on relative importance of input values
cols<-colorRampPalette(c('green','red'))(num.vars)[rank(rel.imp)]

##
#plotting function
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#plot model with new color vector
#separate colors for input vectors using a list for 'circle.col'
plot(mod_BAGSE68475,circle.col=list(cols,'lightblue'))

## ****Deep Learning Net Learning *****************####
library(h2o)
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_BAGSE68475 <- h2o.importFile("dat_BAGSE68475.csv")
dim(train_BAGSE68475)
y_BAGSE68475 = names(train_BAGSE68475)[1]
x_BAGSE68475 = names(train_BAGSE68475)[2:24]
train_BAGSE68475[,y_BAGSE68475] = as.factor(train_BAGSE68475[,y_BAGSE68475])
head(train_BAGSE68475[,1:4])
set.seed(1234)
model_BAGSE68475 = h2o.deeplearning(x=x_BAGSE68475, 
                                     y=y_BAGSE68475, 
                                     training_frame=train_BAGSE68475, 
                                     validation_frame=train_BAGSE68475, 
                                     distribution = "multinomial",
                                     activation = "RectifierWithDropout",
                                     hidden = c(10,10,10,10),
                                     input_dropout_ratio = 0.2,
                                     l1 = 1e-5,
                                     epochs = 50)

print(model_BAGSE68475)

plot(h2o.performance(model_BAGSE68475,valid=T),type='roc',col="blue",lwd=2)
text(0.25,0.75,"BAGSE68475 AUC=1.000")

summary(model_BAGSE68475)
plot(model_BAGSE68475)

# **** Fit DNN ***** ####
library(deepnet)
dim(dat_BAGSE68475)
set.seed(2345)
xx_BAGSE68475=dat_BAGSE68475[,-1]
yy_BAGSE68475=dat_BAGSE68475[,1]
nn_BAGSE68475 <- nn.train(as.matrix(xx_BAGSE68475), yy_BAGSE68475, hidden = c(5))
yyy_BAGSE68475 = nn.predict(nn_BAGSE68475, xx_BAGSE68475)
print(head(yyy_BAGSE68475))
yhat = matrix(0,length(yyy_BAGSE68475),1)
yhat[which(yyy_BAGSE68475 > mean(yyy_BAGSE68475))] = 2
yhat[which(yyy_BAGSE68475<= mean(yyy_BAGSE68475))] = 1
cm_BAGSE68475 = table(yy_BAGSE68475,yhat)
print(cm_BAGSE68475)
print(sum(diag(cm_BAGSE68475))/sum(cm_BAGSE68475))

##################################################
##################################################
######## 2019, Shukla et al. CMIM + BGA ##########
##################################################
#** Scaled Data
GSE34788_dat=read.csv("GSE34788_dat_ScaleX.csv")
dim(GSE34788_dat)
head(GSE34788_dat[1:8])
#CMIM(MadelonD$X,MadelonD$Y,500)
## ***** Step One Filtering *****  ######
## ***** Using CMIM at 99% Cutpoint **###

CMIM=CMIM(GSE34788_dat[,-1],GSE34788_dat[,1],16382)

#GSE34788t.selection <- apply(GSE34788_dat_Scaled[-1], 2, function(x)CMIM=CMIM(GSE34788_dat_Scaled[,-1],GSE34788_dat_Scaled[,1],16382)$score)                                   

GSE34788t.selection <- CMIM$score 
median(GSE34788t.selection)
GSE34788t.selectionCMIM <- sort(GSE34788t.selection[which(GSE34788t.selection >= 0.038688)])
length(GSE34788t.selectionCMIM)
#write.csv(GSE34788t.selectionCMIM,file="GSE34788t.selectionCMIM.csv")
#t.selection=read.csv("t.selection_22.csv")
#GSE34788_dat=read.csv("GSE34788_dat.csv")
cat("\n", "Preliminary selection by t.statistic reporting p.values:", "\n"); utils::flush.console()
GSE34788dat_fil <- subset(GSE34788_dat, select = c("response", names(GSE34788t.selectionCMIM)))
print(length(GSE34788dat_fil))
write.csv(GSE34788dat_fil,"filtered_GSE34788_CMIM.csv")

###****** #####
######### Process 2: Wrapper via  GA ########################

GSE34788_CMIM=read.csv("filtered_GSE34788_CMIM.csv")
head(GSE34788_CMIM[1:4])
trainXGSE34788=GSE34788_CMIM[,-1]
trainyGSE34788=GSE34788_CMIM[,1]
#trainy=ifelse(trainy==1,"Low Heart Rate","High Heart Rate")
yGSE34788=as.factor(trainyGSE34788)
GSE34788_CMIM2=data.frame(yGSE34788,trainXGSE34788)
head(GSE34788_CMIM2[1:4])

set.seed(1234)

registerDoParallel(4) # Registrer a parallel backend for train
getDoParWorkers() # check that there are 4 workers

ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)
## 
set.seed(10)
lev <- c(1,2)     # Set the levels

system.time(rf_gaGSE34788<- gafs(x = trainXGSE34788, y = yGSE34788,
                                 iters = 100, # 100 generations of algorithm
                                 popSize = 20, # population size for each generation
                                 levels = lev,
                                 gafsControl = ga_ctrl))

rf_gaGSE34788
plot(rf_gaGSE34788) # Plot mean fitness (AUC) by generation


final_GAGSE34788 <- rf_gaGSE34788$ga$final # Get features selected by GA
dat_GAGSE34788CMIM <- subset(GSE34788_CMIM2, select = c("yGSE34788", final_GAGSE34788))
head(dat_GAGSE34788CMIM [,1:4])
dim(dat_GAGSE34788CMIM )
write.csv(dat_GAGSE34788CMIM,"dat_GAGSE34788CMIMselected.csv")

## Process 3: Embbeded SVM and KNN ##

head(dat_GAGSE34788CMIM[1:5])
model1 <- svm(yGSE34788 ~ ., data = dat_GAGSE34788CMIM,type="C-classification",gamma=0.005,cost=0.5)
preds <- predict(model1)
confusionMatrix(dat_GAGSE34788CMIM$yGSE34788, preds)

## KNN
library(class)
knnmodel <-  knn(train=dat_GAGSE34788CMIM, test=dat_GAGSE34788CMIM, cl=dat_GAGSE34788CMIM$yGSE34788, k=10)
confusionMatrix(knnmodel,dat_GAGSE34788CMIM$yGSE34788)

## Naive Bayes

nbmodel <- naiveBayes(yGSE34788 ~ ., data = dat_GAGSE34788CMIM)
nbpreds <- predict(nbmodel,dat_GAGSE34788CMIM[,-1])
confusionMatrix(dat_GAGSE34788CMIM$yGSE34788, nbpreds)

############################
### Data GSE86569 ##########

#** Scaled Data
GSE86569_dat=read.csv("GSE86569_dat.csv")
dim(GSE86569_dat)
head(GSE86569_dat[1:4])
#CMIM(MadelonD$X,MadelonD$Y,500)
## ***** Step One Filtering *****  ######
## ***** Using CMIM at 99% Cutpoint **###

CMIM_GSE86569=CMIM(GSE86569_dat[,-1],GSE86569_dat[,1],16382)

#GSE34788t.selection <- apply(GSE34788_dat_Scaled[-1], 2, function(x)CMIM=CMIM(GSE34788_dat_Scaled[,-1],GSE34788_dat_Scaled[,1],16382)$score)                                   

GSE86569t.selection <- CMIM_GSE86569$score 
median(GSE86569t.selection)
GSE86569t.selectionCMIM <- sort(GSE86569t.selection[which(GSE86569t.selection >= 0.02378401)])
length(GSE86569t.selectionCMIM)
#write.csv(GSE34788t.selectionCMIM,file="GSE34788t.selectionCMIM.csv")
#t.selection=read.csv("t.selection_22.csv")
#GSE34788_dat=read.csv("GSE34788_dat.csv")
GSE86569dat_fil <- subset(GSE86569_dat, select = c("response", names(GSE86569t.selectionCMIM)))
print(length(GSE86569dat_fil))
write.csv(GSE86569dat_fil,"filtered_GSE86569_CMIM.csv")

###****** #####
######### Process 2: Wrapper via  GA ########################

GSE86569_CMIM=read.csv("filtered_GSE86569_CMIM.csv")
head(GSE86569_CMIM[1:4])
trainXGSE86569=GSE86569_CMIM[,-1]
trainyGSE86569=GSE86569_CMIM[,1]
#trainy=ifelse(trainy==1,"Low Heart Rate","High Heart Rate")
yGSE86569=as.factor(trainyGSE86569)
GSE86569_CMIM2=data.frame(yGSE86569,trainXGSE86569)
head(GSE86569_CMIM2[1:4])

set.seed(1234)

registerDoParallel(4) # Registrer a parallel backend for train
getDoParWorkers() # check that there are 4 workers

ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)
## 
set.seed(10)
lev <- c(1,2)     # Set the levels

system.time(rf_gaGSE86569<- gafs(x = trainXGSE86569, y = yGSE86569,
                                 iters = 100, # 100 generations of algorithm
                                 popSize = 20, # population size for each generation
                                 levels = lev,
                                 gafsControl = ga_ctrl))

rf_gaGSE86569
plot(rf_gaGSE86569) # Plot mean fitness (AUC) by generation


final_GAGSE86569 <- rf_gaGSE86569$ga$final # Get features selected by GA
dat_GAGSE86569CMIM <- subset(GSE86569_CMIM2, select = c("yGSE86569", final_GAGSE86569))
head(dat_GAGSE86569CMIM [,1:4])
dim(dat_GAGSE86569CMIM )
write.csv(dat_GAGSE86569CMIM,"dat_GAGSE86569CMIMselected.csv")

## Process 3: Embbeded SVM and KNN ##

head(dat_GAGSE86569CMIM[1:5])
model2 <- svm(yGSE86569 ~ ., data = dat_GAGSE86569CMIM,type="C-classification",gamma=0.005,cost=0.5)
preds2 <- predict(model2)
confusionMatrix(dat_GAGSE86569CMIM$yGSE86569, preds2)

## KNN
library(class)
knnmodel2 <-  knn(train=dat_GAGSE86569CMIM, test=dat_GAGSE86569CMIM, cl=dat_GAGSE86569CMIM$yGSE86569, k=10)
confusionMatrix(knnmodel2,dat_GAGSE86569CMIM$yGSE86569)

## N-Bayes
nbmodel2 <- naiveBayes(yGSE86569 ~ ., data = dat_GAGSE86569CMIM)
nbpreds2 <- predict(nbmodel2,dat_GAGSE86569CMIM[,-1])
confusionMatrix(dat_GAGSE86569CMIM$yGSE86569, nbpreds2)


# 3
############################
### Data GSE115574 ##########

#** Scaled Data
GSE115574_dat=read.csv("GSE115574_dat.csv")
dim(GSE115574_dat)
head(GSE115574_dat[1:4])
#CMIM(MadelonD$X,MadelonD$Y,500)
## ***** Step One Filtering *****  ######
## ***** Using CMIM at 99% Cutpoint **###

CMIM_GSE115574=CMIM(GSE115574_dat[,-1],GSE115574_dat[,1],16382)

#GSE34788t.selection <- apply(GSE34788_dat_Scaled[-1], 2, function(x)CMIM=CMIM(GSE34788_dat_Scaled[,-1],GSE34788_dat_Scaled[,1],16382)$score)                                   

GSE115574t.selection <- CMIM_GSE115574$score 
median(GSE115574t.selection)
GSE115574t.selectionCMIM <- sort(GSE115574t.selection[which(GSE115574t.selection >= 0.08080466)])
length(GSE115574t.selectionCMIM)
#write.csv(GSE34788t.selectionCMIM,file="GSE34788t.selectionCMIM.csv")
#t.selection=read.csv("t.selection_22.csv")
#GSE34788_dat=read.csv("GSE34788_dat.csv")
GSE115574dat_fil <- subset(GSE115574_dat, select = c("response", names(GSE115574t.selectionCMIM)))
print(length(GSE115574dat_fil))
write.csv(GSE115574dat_fil,"filtered_GSE115574_CMIM.csv")

###****** #####
######### Process 2: Wrapper via  GA ########################

GSE115574_CMIM=read.csv("filtered_GSE115574_CMIM.csv")
head(GSE115574_CMIM[1:4])
trainXGSE115574=GSE115574_CMIM[,-1]
trainyGSE115574=GSE115574_CMIM[,1]
#trainy=ifelse(trainy==1,"Low Heart Rate","High Heart Rate")
yGSE115574=as.factor(trainyGSE115574)
GSE115574_CMIM2=data.frame(yGSE115574,trainXGSE115574)
head(GSE115574_CMIM2[1:4])

set.seed(1234)

registerDoParallel(4) # Registrer a parallel backend for train
getDoParWorkers() # check that there are 4 workers

ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)
## 
set.seed(10)
lev <- c(1,2)     # Set the levels

system.time(rf_gaGSE115574<- gafs(x = trainXGSE115574, y = yGSE115574,
                                 iters = 100, # 100 generations of algorithm
                                 popSize = 20, # population size for each generation
                                 levels = lev,
                                 gafsControl = ga_ctrl))

rf_gaGSE115574
plot(rf_gaGSE115574) # Plot mean fitness (AUC) by generation


final_GAGSE115574 <- rf_gaGSE115574$ga$final # Get features selected by GA
dat_GAGSE115574CMIM <- subset(GSE115574_CMIM2, select = c("yGSE115574", final_GAGSE115574))
head(dat_GAGSE115574CMIM [,1:4])
dim(dat_GAGSE115574CMIM )
write.csv(dat_GAGSE115574CMIM,"dat_GAGSE115574CMIMselected.csv")

## Process 3: Embbeded SVM and KNN ##

head(dat_GAGSE115574CMIM[1:5])
model3 <- svm(yGSE115574 ~ ., data = dat_GAGSE115574CMIM,type="C-classification",gamma=0.005,cost=0.5)
preds3 <- predict(model3,dat_GAGSE115574CMIM[,-1])
confusionMatrix(dat_GAGSE115574CMIM$yGSE115574, preds3)

## KNN
library(class)
knnmodel3 <-  knn(train=dat_GAGSE115574CMIM, test=dat_GAGSE115574CMIM, cl=dat_GAGSE115574CMIM$yGSE115574, k=10)
confusionMatrix(knnmodel3,dat_GAGSE115574CMIM$yGSE115574)

## N-Bayes
nbmodel3 <- naiveBayes(yGSE115574 ~ ., data = dat_GAGSE115574CMIM)
nbpreds3 <- predict(nbmodel3,dat_GAGSE115574CMIM[,-1])
confusionMatrix(dat_GAGSE115574CMIM$yGSE115574, nbpreds3)

## 4

############################
### Data GSE112266 ##########

#** Scaled Data
GSE112266_dat=read.csv("GSE112266_dat.csv")
dim(GSE112266_dat)
head(GSE112266_dat[1:4])
#CMIM(MadelonD$X,MadelonD$Y,500)
## ***** Step One Filtering *****  ######
## ***** Using CMIM at 99% Cutpoint **###

CMIM_GSE112266=CMIM(GSE112266_dat[,-1],GSE112266_dat[,1],107)

#GSE34788t.selection <- apply(GSE34788_dat_Scaled[-1], 2, function(x)CMIM=CMIM(GSE34788_dat_Scaled[,-1],GSE34788_dat_Scaled[,1],16382)$score)                                   

GSE112266t.selection <- CMIM_GSE112266$score 
mean(GSE112266t.selection)
GSE112266t.selectionCMIM <- sort(GSE112266t.selection[which(GSE112266t.selection >= 0.07820419)])
length(GSE112266t.selectionCMIM)
#write.csv(GSE34788t.selectionCMIM,file="GSE34788t.selectionCMIM.csv")
#t.selection=read.csv("t.selection_22.csv")
#GSE34788_dat=read.csv("GSE34788_dat.csv")
GSE112266dat_fil <- subset(GSE112266_dat, select = c("response", names(GSE112266t.selectionCMIM)))
print(length(GSE112266dat_fil))
write.csv(GSE112266dat_fil,"filtered_GSE112266_CMIM.csv")

###****** #####
######### Process 2: Wrapper via  GA ########################

GSE112266_CMIM=read.csv("filtered_GSE112266_CMIM.csv")
head(GSE112266_CMIM[1:4])
trainXGSE112266=GSE112266_CMIM[,-1]
trainyGSE112266=GSE112266_CMIM[,1]
#trainy=ifelse(trainy==1,"Low Heart Rate","High Heart Rate")
yGSE112266=as.factor(trainyGSE112266)
GSE112266_CMIM2=data.frame(yGSE112266,trainXGSE112266)
head(GSE112266_CMIM2[1:4])

set.seed(1234)

registerDoParallel(4) # Registrer a parallel backend for train
getDoParWorkers() # check that there are 4 workers

ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)
## 
set.seed(10)
lev <- c(1,2)     # Set the levels

system.time(rf_gaGSE112266<- gafs(x = trainXGSE112266, y = yGSE112266,
                                  iters = 100, # 100 generations of algorithm
                                  popSize = 20, # population size for each generation
                                  levels = lev,
                                  gafsControl = ga_ctrl))

rf_gaGSE112266
plot(rf_gaGSE112266) # Plot mean fitness (AUC) by generation


final_GAGSE112266 <- rf_gaGSE112266$ga$final # Get features selected by GA
dat_GAGSE112266CMIM <- subset(GSE112266_CMIM2, select = c("yGSE112266", final_GAGSE112266))
head(dat_GAGSE112266CMIM [,1:4])
dim(dat_GAGSE112266CMIM )
write.csv(dat_GAGSE112266CMIM,"dat_GAGSE112266CMIMselected.csv")

## Process 3: Embbeded SVM and KNN ##

head(dat_GAGSE112266CMIM[1:3])
model4 <- svm(yGSE112266 ~ ., data = dat_GAGSE112266CMIM,type="C-classification",gamma=0.005,cost=0.5)
preds4 <- predict(model4,dat_GAGSE112266CMIM[,-1])
confusionMatrix(dat_GAGSE112266CMIM$yGSE112266, preds4)

## KNN
library(class)
knnmodel4 <-  knn(train=dat_GAGSE112266CMIM, test=dat_GAGSE112266CMIM, cl=dat_GAGSE112266CMIM$yGSE112266, k=10)
confusionMatrix(knnmodel4,dat_GAGSE112266CMIM$yGSE112266)

## N-Bayes
nbmodel4 <- naiveBayes(yGSE112266 ~ ., data = dat_GAGSE112266CMIM)
nbpreds4 <- predict(nbmodel4,dat_GAGSE112266CMIM[,-1])
confusionMatrix(dat_GAGSE112266CMIM$yGSE112266, nbpreds4)

## 5

############################
### Data GSE68475 ##########

#** Scaled Data
GSE68475_dat=read.csv("GSE68475_dat.csv")
dim(GSE68475_dat)
head(GSE68475_dat[1:4])
#CMIM(MadelonD$X,MadelonD$Y,500)
## ***** Step One Filtering *****  ######
## ***** Using CMIM at 99% Cutpoint **###

CMIM_GSE68475=CMIM(GSE68475_dat[,-1],GSE68475_dat[,1],16382)

#GSE34788t.selection <- apply(GSE34788_dat_Scaled[-1], 2, function(x)CMIM=CMIM(GSE34788_dat_Scaled[,-1],GSE34788_dat_Scaled[,1],16382)$score)                                   

GSE68475t.selection <- CMIM_GSE68475$score 
median(GSE68475t.selection)
GSE68475t.selectionCMIM <- sort(GSE68475t.selection[which(GSE68475t.selection >= 0.02491658)])
length(GSE68475t.selectionCMIM)
#write.csv(GSE34788t.selectionCMIM,file="GSE34788t.selectionCMIM.csv")
#t.selection=read.csv("t.selection_22.csv")
#GSE34788_dat=read.csv("GSE34788_dat.csv")
GSE68475dat_fil <- subset(GSE68475_dat, select = c("response", names(GSE68475t.selectionCMIM)))
print(length(GSE68475dat_fil))
write.csv(GSE68475dat_fil,"filtered_GSE68475_CMIM.csv")

###****** #####
######### Process 2: Wrapper via  GA ########################

GSE68475_CMIM=read.csv("filtered_GSE68475_CMIM.csv")
head(GSE68475_CMIM[1:4])
trainXGSE68475=GSE68475_CMIM[,-1]
trainyGSE68475=GSE68475_CMIM[,1]
#trainy=ifelse(trainy==1,"Low Heart Rate","High Heart Rate")
yGSE68475=as.factor(trainyGSE68475)
GSE68475_CMIM2=data.frame(yGSE68475,trainXGSE68475)
head(GSE68475_CMIM2[1:4])

set.seed(1234)

registerDoParallel(4) # Registrer a parallel backend for train
getDoParWorkers() # check that there are 4 workers

ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)
## 
set.seed(10)
lev <- c(1,2)     # Set the levels

system.time(rf_gaGSE68475<- gafs(x = trainXGSE68475, y = yGSE68475,
                                  iters = 100, # 100 generations of algorithm
                                  popSize = 20, # population size for each generation
                                  levels = lev,
                                  gafsControl = ga_ctrl))

rf_gaGSE68475
plot(rf_gaGSE68475) # Plot mean fitness (AUC) by generation


final_GAGSE68475 <- rf_gaGSE68475$ga$final # Get features selected by GA
dat_GAGSE68475CMIM <- subset(GSE68475_CMIM2, select = c("yGSE68475", final_GAGSE68475))
head(dat_GAGSE68475CMIM [,1:4])
dim(dat_GAGSE68475CMIM )
write.csv(dat_GAGSE68475CMIM,"dat_GAGSE68475CMIMselected.csv")

## Process 3: Embbeded SVM and KNN ##

head(dat_GAGSE68475CMIM[1:5])
model5 <- svm(yGSE68475 ~ ., data = dat_GAGSE68475CMIM,type="C-classification",gamma=0.005,cost=0.5)
preds5 <- predict(model5,dat_GAGSE68475CMIM[,-1])
confusionMatrix(dat_GAGSE68475CMIM$yGSE68475, preds5)

## KNN
library(class)
knnmodel5 <-  knn(train=dat_GAGSE68475CMIM, test=dat_GAGSE68475CMIM, cl=dat_GAGSE68475CMIM$yGSE68475, k=10)
confusionMatrix(knnmodel5,dat_GAGSE68475CMIM$yGSE68475)

## N-Bayes
nbmodel5 <- naiveBayes(yGSE68475 ~ ., data = dat_GAGSE68475CMIM)
nbpreds5 <- predict(nbmodel5,dat_GAGSE68475CMIM[,-1])
confusionMatrix(dat_GAGSE68475CMIM$yGSE68475, nbpreds5)


