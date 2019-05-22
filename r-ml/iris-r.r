# To execute in VSCode term:
# & "C:/Program Files/R/R-3.6.0/bin/x64/Rscript.exe" c:/dev/ml-prototyping/r-ml/iris-r.r

# Install in elevated terminal first
# install.packages("caret")

# Load dependencies
library(caret)

# Load training dataset
data(iris)
dataset <- iris

# Create validation dataset from training dataset
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
validation <- dataset[-validation_index,]
dataset <- dataset[validation_index,]

### Test algoritms for accuracy ###

control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# Linear Discriminant Analysis
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
# Classification and Regression Trees
#set.seed(7)
#fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
# k-Nearest Neighbor
#set.seed(7)
#fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
# Support Vector Machines
#set.seed(7)
#fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
#set.seed(7)
#fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

# Summarize algorithm comparison
#results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
#summary(results)

# Use validation dataset to make predictions
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

# Make single prediction
single <- list(5.5, 2.4, 3.8, 1.1, "versicolor")
names(single) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")
single_predict <- predict(fit.lda, single)
paste("Single Prediction Result: ", single_predict)