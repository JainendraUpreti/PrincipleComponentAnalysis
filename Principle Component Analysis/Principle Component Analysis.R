library(tidyr)
library(dplyr)
library(knitr)
library(ggplot2)
library(RCurl) # To download data from URL
library(caret)
library(tibble)
library(xgboost)
install.packages("caret")
urlfile <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data'
x <- getURL(urlfile, ssl.verifypeer = FALSE)
gisetteRaw <- read.table(textConnection(x), sep = '', header = FALSE, stringsAsFactors = FALSE)

urlfile <- "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels"
x <- getURL(urlfile, ssl.verifypeer = FALSE)
g_labels <- read.table(textConnection(x), sep = '', header = FALSE, stringsAsFactors = FALSE)

print(dim(gisetteRaw))
head(gisetteRaw)

#-------------------------------------------------------------------------------------------------#
# PCA creates new features using the many features present in the dataset. A problem that may     #
# arise with PCA is that if the variables have zero variance or very little variance, i.e., almost#
# the same values throughtout the observations, the PCA collapses. We use caret package to check  #
# that using the zero variance funtion.
#-------------------------------------------------------------------------------------------------#

nzv <- nearZeroVar(gisetteRaw, saveMetrics = TRUE) #SaveMetrics options gives us the amount of %age
                                                   # of zero variance of each feature
?nearZeroVar
print(paste('Range:', range(nzv$percentUnique)))
print(paste('Column count before cutoff:',ncol(gisetteRaw)))
dim(nzv)

head(nzv) # in the output PercentUnique tells us how much percent uniqueness is there in a variable
          # Does it has zero variance and nzv tells us if it is close to near zero variance

# we get a range from 0 to 8.6 for Percent unique, based on it we choose a cutoff of 0.1

dim(nzv[nzv$percentUnique > 0.1,]) # from 5000 to 4639; 361 columns eliminated
gisette_nzv <- gisetteRaw[c(rownames(nzv[nzv$percentUnique > 0.1,]))] 
#rownames() functions passes the name of columns to be used as rownames for our original dataset
print(paste('Column count after cutoff:',ncol(gisette_nzv)))

# after cleaning up we check how it is performing without PCA transformation
# for this we have chosen xgboost since the dimensionality is huge and normal model won't work

# function for xgboost
EvaluateAUC <- function(dfEvaluate) {
  require(xgboost)
  require(Metrics)
  CVs <- 5
  cvDivider <- floor(nrow(dfEvaluate) / (CVs+1))
  indexCount <- 1
  outcomeName <- c('cluster')
  predictors <- names(dfEvaluate)[!names(dfEvaluate) %in% outcomeName]
  lsErr <- c()
  lsAUC <- c()
  for (cv in seq(1:CVs)) {
    print(paste('cv',cv))
    dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
    dataTest <- dfEvaluate[dataTestIndex,]
    dataTrain <- dfEvaluate[-dataTestIndex,]
    
    bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                   label = dataTrain[,outcomeName],
                   max.depth=6, eta = 1, verbose=0,
                   nround=5, nthread=4, 
                   objective = "reg:linear")
    
    predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
    err <- rmse(dataTest[,outcomeName], predictions)
    auc <- auc(dataTest[,outcomeName],predictions)
    
    lsErr <- c(lsErr, err)
    lsAUC <- c(lsAUC, auc)
    gc()
  }
  print(paste('Mean Error:',mean(lsErr)))
  print(paste('Mean AUC:',mean(lsAUC)))
}

dfEvaluate <- cbind(as.data.frame(sapply(gisette_nzv, as.numeric)),
                    cluster=g_labels$V1)
EvaluateAUC(dfEvaluate) # we get 0.97, now using PCA we want to get as close as possible to this

# Preparing data for PCA

pmatrix <- scale(gisette_nzv)
princ <- prcomp(pmatrix)

