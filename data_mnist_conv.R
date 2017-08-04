# get data into format

library(keras)

mnist <- dataset_mnist()

###

X_train <- mnist$train$x/255
dim(X_train)
dim(X_train) <- c(dim(X_train),1)
dim(X_train)

###

X_test <- mnist$test$x/255
dim(X_test) <- c(dim(X_test),1)


###

img_rows <- 28L
img_cols <- 28L
img_chns <- 1L