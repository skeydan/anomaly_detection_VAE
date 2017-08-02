library(keras)

mnist <- dataset_mnist()
data <- lapply(mnist, function(m) {
  array(m$x / 255, dim = c(dim(m$x)[1], original_img_size))
})
X_train <- data$train
X_test <- data$test

dim(X_train)
dim(X_test)