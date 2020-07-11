#Practical Machine Learning Project : Prediction Assignment Writeup

##I. Overview
This document is the final report of the Peer Assessment project from Coursera’s course Practical Machine Learning, as part of the Specialization in Data Science. It was built up in RStudio, using its knitr functions, meant to be published in html format. This analysis meant to be the basis for the course quiz and a prediction assignment writeup. The main goal of the project is to predict the manner in which 6 participants performed some exercise as described below. This is the “classe” variable in the training set. The machine learning algorithm described here is applied to the 20 test cases available in the test data and the predictions are submitted in appropriate format to the Course Project Prediction Quiz for automated grading.

##II. Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

##Machine Learning - Step by Step process (7 Steps)
a) Data Collection (Loading)
b) Data Preparation (Cleaning)
c) Choosing a model
d) Training the Model
e) Evaluate the Model
f) Parameter Tuning
g) Make Predictions

##III. Data Loading and Exploratory Analysis

#a) Data Collection (Loading)
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.
```{r setup, include=TRUE}
rm(list=ls())                # free up memory for download of the data sets
knitr::opts_chunk$set(echo = TRUE)
```
##Environment Preparation
We first upload the R libraries that are necessary for the complete analysis.Setting seed, loading libraries and dataset.
```{r, include=T, echo=T,results=F, message=F}
set.seed(1967) 
library(knitr)
library(lattice)
library(ggplot2)
#install.packages("caret", dependencies = TRUE)
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(data.table)
library(corrplot)
library(plotly)
library(RGtk2)
library(gbm)
```

Set working directory
```{r}
setwd("~/0JohnsHopkins/8-PracticalMachineLearning/Course Project I")
```

Load the training and test datasets
The next step is loading the data set. The training dataset is then partinioned in 2 to create a Training set (70% of the data) for the modeling process and a Test set (with the remaining 30%) for the validations. The testing data set is not changed and will only be used for the quiz results generation.
```{r}
#command to read trainig and test set
data_train <- read.csv("pml-training.csv")
data_quiz <- read.csv("pml-testing.csv")
dim(data_train)
dim(data_quiz)
```
Both created datasets have 160 variables. Those variables have plenty of NA, that can be removed with the cleaning procedures below. 

```{r}
in_train  <- createDataPartition(data_train$classe, p=0.75, list=FALSE)
train_set <- data_train[ in_train, ]
test_set  <- data_train[-in_train, ]
dim(train_set)
dim(test_set)
```

#b) Data Preparation (Cleaning)
The Near Zero variance (NZV) variables are also removed and the ID variables as well.
```{r}
nzv_var <- nearZeroVar(train_set)
train_set <- train_set[ , -nzv_var]
test_set  <- test_set [ , -nzv_var]
dim(train_set)
dim(test_set)
```
```{r}
#Remove variables that are mostly NA. A threshlod of 95 % is selected
na_var <- sapply(train_set, function(x) mean(is.na(x))) > 0.95
train_set <- train_set[ , na_var == FALSE]
test_set  <- test_set [ , na_var == FALSE]
dim(train_set)
dim(test_set)
```
```{r}
#Since columns 1 to 5 are identification variables only, they will be removed as well
train_set <- train_set[ , -(1:5)]
test_set  <- test_set [ , -(1:5)]
dim(train_set)
dim(test_set)
```

##Correlation Analysis
A correlation among variables could be analyzed before proceeding to the modeling procedures.
```{r}
corr_matrix <- cor(train_set[ , -54])
corrplot(corr_matrix, order = "FPC", method = "circle", type = "lower",
         tl.cex = 0.6, tl.col = rgb(0, 0, 0))
```
The number of variables for the analysis has been reduced from the original 160 down to 54

#c) Choosing a model
Three methods will be applied to model the regressions (in the Train data set) and the best one (with higher accuracy when applied to the Test data set) will be used for the quiz predictions. 
The methods are: 
- Decision Tree, 
- Generalized Boosted Model (GBM) and 
- Random Forests (rf), as described below. 

#d) Training the Model
A Confusion Matrix is plotted at the end of each analysis to better visualize the accuracy of the models.

#Decision Tree Model
```{r}
set.seed(1967)
fit_DT <- rpart(classe ~ ., data = train_set, method="class")
fancyRpartPlot(fit_DT)
```

```{r}
predict_DT <- predict(fit_DT, newdata = test_set, type="class")
conf_matrix_DT <- confusionMatrix(table(predict_DT, test_set$classe))
conf_matrix_DT
```

```{r}
plot(conf_matrix_DT$table, col = conf_matrix_DT$byClass, 
     main = paste("Decision Tree Model: Predictive Accuracy =",
                  round(conf_matrix_DT$overall['Accuracy'], 4)))
```

#Generalized Boosted Model (GBM)
```{r}
set.seed(1967)
ctrl_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_GBM  <- train(classe ~ ., data = train_set, method = "gbm",
                  trControl = ctrl_GBM, verbose = FALSE)
fit_GBM$finalModel
```

```{r}
predict_GBM <- predict(fit_GBM, newdata = test_set)
conf_matrix_GBM <- confusionMatrix(table(predict_GBM, test_set$classe))
conf_matrix_GBM
```

#Random Forest
```{r}
set.seed(1967)
ctrl_RF <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_RF  <- train(classe ~ ., data = train_set, method = "rf",
                 trControl = ctrl_RF, verbose = FALSE)
fit_RF$finalModel
```

```{r}
predict_RF <- predict(fit_RF, newdata = test_set)
conf_matrix_RF <- confusionMatrix(table(predict_RF, test_set$classe))
conf_matrix_RF
```

#e) Evaluate the Model (Fitting models)

#Applying the Best Predictive Model to the Test Data
To summarize, the predictive accuracy of the three models evaluated is as follows:

Summary of the results:
- Decision tree model     - is the worst model running, has the low mean and the highest standard deviation.
- GBM model               - has a decent mean accuracy but a little bit lower accuracy than RF,
- Random Fores model      - has the highest mean accuracy and lowest standard deviation

#f) Parameter Tuning
Checking prediction accuracy on my own testing/validation set. I am expecting similar accuracy as the mean from the cross validation.

##The kappa statistic 
The kappa statistic (labeled Kappa in the previous output) adjusts accuracy by accounting for the possibility of a correct prediction by chance alone. Kappa values range to a maximum value of 1, which indicates perfect agreement between the model's predictions and the true values—a rare occurrence. Values less than one indicate imperfect agreement. 

Depending on how your model is to be used, the interpretation of the kappa statistic might vary. One common interpretation is shown as follows: 
• Poor agreement = Less than 0.20 
• Fair agreement = 0.20 to 0.40 
• Moderate agreement = 0.40 to 0.60 
• Good agreement = 0.60 to 0.80 
• Very good agreement = 0.80 to 1.00

This three models preforms as expected, the deviation from the cross validation accuracy is low and I do not see a reason to change resampling method or adding repetitons.

Checking if there are anything to gain from increasing the number of boosting iterations.
```{r}
plot(fit_RF)
print(fit_RF$bestTune)
```

The predictive accuracy of the Random Forest model is excellent at 99.8 %.
Accuracy has plateaued, and further tuning would only yield decimal gain. 
- The best tuning parameters hads 150 trees (boosting iterations), 
- interaction depth 3
- shrinkage 0.1. 

#g) Make Predictions
Deciding to predict with this model.

Decision Tree Model: 72.0 %
Generalized Boosted Model: 98.86 %
Random Forest Model: 99.78 %

The Random Forest model is selected and applied to make predictions on the 20 data points from the original testing dataset (data_quiz).
```{r}
cat("Predictions: ", paste(predict(fit_RF, data_quiz)))
=======
#Practical Machine Learning Project : Prediction Assignment Writeup

##I. Overview
This document is the final report of the Peer Assessment project from Coursera’s course Practical Machine Learning, as part of the Specialization in Data Science. It was built up in RStudio, using its knitr functions, meant to be published in html format. This analysis meant to be the basis for the course quiz and a prediction assignment writeup. The main goal of the project is to predict the manner in which 6 participants performed some exercise as described below. This is the “classe” variable in the training set. The machine learning algorithm described here is applied to the 20 test cases available in the test data and the predictions are submitted in appropriate format to the Course Project Prediction Quiz for automated grading.

##II. Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

##Machine Learning - Step by Step process (7 Steps)
a) Data Collection (Loading)
b) Data Preparation (Cleaning)
c) Choosing a model
d) Training the Model
e) Evaluate the Model
f) Parameter Tuning
g) Make Predictions

##III. Data Loading and Exploratory Analysis

#a) Data Collection (Loading)
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.
```{r setup, include=TRUE}
rm(list=ls())                # free up memory for download of the data sets
knitr::opts_chunk$set(echo = TRUE)
```
##Environment Preparation
We first upload the R libraries that are necessary for the complete analysis.Setting seed, loading libraries and dataset.
```{r, include=T, echo=T,results=F, message=F}
set.seed(1967) 
library(knitr)
library(lattice)
library(ggplot2)
#install.packages("caret", dependencies = TRUE)
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(data.table)
library(corrplot)
library(plotly)
library(RGtk2)
library(gbm)
```

Set working directory
```{r}
setwd("~/0JohnsHopkins/8-PracticalMachineLearning/Course Project I")
```

Load the training and test datasets
The next step is loading the data set. The training dataset is then partinioned in 2 to create a Training set (70% of the data) for the modeling process and a Test set (with the remaining 30%) for the validations. The testing data set is not changed and will only be used for the quiz results generation.
```{r}
#command to read trainig and test set
data_train <- read.csv("pml-training.csv")
data_quiz <- read.csv("pml-testing.csv")
dim(data_train)
dim(data_quiz)
```
Both created datasets have 160 variables. Those variables have plenty of NA, that can be removed with the cleaning procedures below. 

```{r}
in_train  <- createDataPartition(data_train$classe, p=0.75, list=FALSE)
train_set <- data_train[ in_train, ]
test_set  <- data_train[-in_train, ]
dim(train_set)
dim(test_set)
```

#b) Data Preparation (Cleaning)
The Near Zero variance (NZV) variables are also removed and the ID variables as well.
```{r}
nzv_var <- nearZeroVar(train_set)
train_set <- train_set[ , -nzv_var]
test_set  <- test_set [ , -nzv_var]
dim(train_set)
dim(test_set)
```
```{r}
#Remove variables that are mostly NA. A threshlod of 95 % is selected
na_var <- sapply(train_set, function(x) mean(is.na(x))) > 0.95
train_set <- train_set[ , na_var == FALSE]
test_set  <- test_set [ , na_var == FALSE]
dim(train_set)
dim(test_set)
```
```{r}
#Since columns 1 to 5 are identification variables only, they will be removed as well
train_set <- train_set[ , -(1:5)]
test_set  <- test_set [ , -(1:5)]
dim(train_set)
dim(test_set)
```

##Correlation Analysis
A correlation among variables could be analyzed before proceeding to the modeling procedures.
```{r}
corr_matrix <- cor(train_set[ , -54])
corrplot(corr_matrix, order = "FPC", method = "circle", type = "lower",
         tl.cex = 0.6, tl.col = rgb(0, 0, 0))
```
The number of variables for the analysis has been reduced from the original 160 down to 54

#c) Choosing a model
Three methods will be applied to model the regressions (in the Train data set) and the best one (with higher accuracy when applied to the Test data set) will be used for the quiz predictions. 
The methods are: 
- Decision Tree, 
- Generalized Boosted Model (GBM) and 
- Random Forests (rf), as described below. 

#d) Training the Model
A Confusion Matrix is plotted at the end of each analysis to better visualize the accuracy of the models.

#Decision Tree Model
```{r}
set.seed(1967)
fit_DT <- rpart(classe ~ ., data = train_set, method="class")
fancyRpartPlot(fit_DT)
```

```{r}
predict_DT <- predict(fit_DT, newdata = test_set, type="class")
conf_matrix_DT <- confusionMatrix(table(predict_DT, test_set$classe))
conf_matrix_DT
```

```{r}
plot(conf_matrix_DT$table, col = conf_matrix_DT$byClass, 
     main = paste("Decision Tree Model: Predictive Accuracy =",
                  round(conf_matrix_DT$overall['Accuracy'], 4)))
```

#Generalized Boosted Model (GBM)
```{r}
set.seed(1967)
ctrl_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_GBM  <- train(classe ~ ., data = train_set, method = "gbm",
                  trControl = ctrl_GBM, verbose = FALSE)
fit_GBM$finalModel
```

```{r}
predict_GBM <- predict(fit_GBM, newdata = test_set)
conf_matrix_GBM <- confusionMatrix(table(predict_GBM, test_set$classe))
conf_matrix_GBM
```

#Random Forest
```{r}
set.seed(1967)
ctrl_RF <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_RF  <- train(classe ~ ., data = train_set, method = "rf",
                 trControl = ctrl_RF, verbose = FALSE)
fit_RF$finalModel
```

```{r}
predict_RF <- predict(fit_RF, newdata = test_set)
conf_matrix_RF <- confusionMatrix(table(predict_RF, test_set$classe))
conf_matrix_RF
```

#e) Evaluate the Model (Fitting models)

#Applying the Best Predictive Model to the Test Data
To summarize, the predictive accuracy of the three models evaluated is as follows:

Summary of the results:
- Decision tree model     - is the worst model running, has the low mean and the highest standard deviation.
- GBM model               - has a decent mean accuracy but a little bit lower accuracy than RF,
- Random Fores model      - has the highest mean accuracy and lowest standard deviation

#f) Parameter Tuning
Checking prediction accuracy on my own testing/validation set. I am expecting similar accuracy as the mean from the cross validation.

##The kappa statistic 
The kappa statistic (labeled Kappa in the previous output) adjusts accuracy by accounting for the possibility of a correct prediction by chance alone. Kappa values range to a maximum value of 1, which indicates perfect agreement between the model's predictions and the true values—a rare occurrence. Values less than one indicate imperfect agreement. 

Depending on how your model is to be used, the interpretation of the kappa statistic might vary. One common interpretation is shown as follows: 
• Poor agreement = Less than 0.20 
• Fair agreement = 0.20 to 0.40 
• Moderate agreement = 0.40 to 0.60 
• Good agreement = 0.60 to 0.80 
• Very good agreement = 0.80 to 1.00

This three models preforms as expected, the deviation from the cross validation accuracy is low and I do not see a reason to change resampling method or adding repetitons.

Checking if there are anything to gain from increasing the number of boosting iterations.
```{r}
plot(fit_RF)
print(fit_RF$bestTune)
```

The predictive accuracy of the Random Forest model is excellent at 99.8 %.
Accuracy has plateaued, and further tuning would only yield decimal gain. 
- The best tuning parameters hads 150 trees (boosting iterations), 
- interaction depth 3
- shrinkage 0.1. 

#g) Make Predictions
Deciding to predict with this model.

Decision Tree Model: 72.0 %
Generalized Boosted Model: 98.86 %
Random Forest Model: 99.78 %

The Random Forest model is selected and applied to make predictions on the 20 data points from the original testing dataset (data_quiz).
```{r}
cat("Predictions: ", paste(predict(fit_RF, data_quiz)))
>>>>>>> abefae462da2bda6c064e0fcc0043ed00df665fb
```
