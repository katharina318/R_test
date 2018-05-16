#library(tidyverse)
#library(lattice)
#library(caret)
#library(caretEnsemble)
#library(purrr)
library(MASS)
library(SuperLearner)

data(Boston, package = 'MASS')
colSums(is.na(Boston))
outcome=Boston$medv
data=subset(Boston, select = -medv)
str(data)
dim(data)

set.seed(1)

train_obs= sample(nrow(data), 150)

X_train=data[train_obs,]
X_holdout=data[-train_obs,]

outcome_bin=as.numeric(outcome>22)

Y_train= outcome_bin[train_obs]
Y_holdout=outcome_bin[-train_obs]

table(Y_train, useNA='ifany')

#available models
listWrappers()

#peek at model code
SL.glmnet

library(glmnet)
library(Matrix)
library(foreach)

# Lasso
set.seed(1)
sl_lasso = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = 'SL.glmnet')

names(sl_lasso)

#risk of best model (discrete SuperLearner winner)
sl_lasso$cvRisk[which.min(sl_lasso$cvRisk)]

#raw glmnet result object
str(sl_lasso$fitLibrary$SL.glmnet_All$object, max.level=1)

library(randomForest)
sl_rf = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = 'SL.randomForest')

sl_rf
SL.randomForest_All

set.seed(1)
sl = SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = c('SL.mean', 'SL.glmnet', 'SL.randomForest'))
sl

#how long did it take to run SL
sl$times$everything

pred = predict(sl, X_holdout, onlySL = T)
str(pred)

summary(pred$library.predict)
library(ggplot2)

qplot(pred$pred[,1])+theme_minimal()
qplot(Y_holdout, pred$pred[,1])+theme_minimal()

#AUC
pred_rocr = ROCR::prediction(pred$pred, Y_holdout)
auc = ROCR::performance(pred_rocr, measure = 'auc', x.measure = 'cutoff')@y.values[[1]]
auc

#ensemble croos-validation
set.seed(1)
system.time({cv_s1 = CV.SuperLearner(Y=Y_train, X =X_train, family = binomial(), V = 3, SL.library = c('SL.mean', 'SL.glmnet', 'SL.randomForest'))})
summary(cv_s1)

#distribution of best single learner as external CV folds
table(simplify2array(cv_s1$whichDiscreteSL))
plot(cv_s1)+theme_classic()

#customize model hyperparameter
#1
SL.randomForest

SL.rf.better = function(...) {SL.randomForest(..., ntree=3000)}

set.seed(1)

cv_s1 = CV.SuperLearner(Y = Y_train, X=X_train, family = binomial(), V=3, SL.library = c('SL.mean', 'SL.glmnet', 'SL.rf.better', 'SL.randomForest'))
summary(cv_s1)

#2
learners = create.Learner('SL.random.Forest', params = list(ntree=3000))
learners
learners$names
SL.random.Forest_1

set.seed(1)
cv_s1 = CV.SuperLearner(Y=Y_train, X=X_train, family = binomial(), V=3, SL.library= c('SL.mean', 'SL.glmnet', learners$names, 'SL.randomForest'))
summary(cv_s1)
plot(cv_s1)+theme_get()

#for RF 2 hyperparameters = particularly important: mtry (nr. of features randomly chosen within each node) & max leaf nodes
#try 3 mtry options

#sqrt(p):default value of mtry for classification
floor(sqrt(ncol(X_train)))
#multiply default by 0.5, 1 & 2
mtry_seq = floor(sqrt(ncol(X_train))*c(0.5, 1, 2))
mtry_seq

learners = create.Learner('SL.randomForest', tune =list(mtry=mtry_seq))
learners

SL.randomForest_1
SL.randomForest_2
SL.randomForest_3

set.seed(1)

cv_s1 = CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), V = 3, SL.library = c('SL.mean', 'SL.glmnet', learners$names, 'SL.randomForest'))
summary(cv_s1)

#Multicore computation
#set up 'snow' system

num_cores = RhpcBLASctl::get_num_cores()
num_cores

cluster = parallel::makeCluster(2)
cluster
parallel::clusterEvalQ(cluster, library(SuperLearner))
parallel::clusterExport(cluster, learners$names)
#set seed for all clusters
library(parallel)
parallel::clusterSetRNGStream(cluster, 1)
system.time({cv_s1 =CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), V = 10, parallel = cluster, SL.library = c('SL.mean', 'SL.glmnet', learners$names, 'SL.randomForest'))})
summary(cv_s1)
#stop cluster when done
parallel::stopCluster(cluster)

