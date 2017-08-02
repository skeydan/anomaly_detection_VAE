library(keras)

mnist <- dataset_mnist()
X_train <- mnist$train$x/255
dim(X_train)
X_test <- mnist$test$x/255
X_train <- X_train %>% apply(1, as.numeric) %>% t()
dim(X_train)
X_test <- X_test %>% apply(1, as.numeric) %>% t()