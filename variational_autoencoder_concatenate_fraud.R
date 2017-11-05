library(ggplot2)
library(dplyr)
library(keras)
(K <- keras::backend())

set.seed(777)

source("data_fraud_concatenate.R")

### general ###
model_weights_exist <- TRUE
weights_file <- "weights_fraud_concatenate_embeddings10_intermediate4.h5"

### dimensions ### 
original_dim <- 3L
latent_dim <- 2L        
intermediate_dim <- 4L 

num_categories <- 4548
num_numeric <- 2
embedding_dim <- 10
input_length <- 1

### hyperparameters ###
l1 <- 0
l2 <- 0.0
batch_size <- 100L
epochs <- 100L
learning_rate <- 0.001

### model ###
zmean_activation <- "linear"
zlogvar_activation <- "relu"
decoded_mean_activation <- "linear"

epsilon_std <- 1.0

### model
x_categorical <- layer_input(shape = 1, name = "categorical_input")
x_embedding <- layer_embedding(input_dim = num_categories,
                               output_dim = embedding_dim,
                               input_length = input_length)(x_categorical)
x_embedding
x_reshape <-
  layer_reshape(target_shape = embedding_dim,
                input_shape = c(1, embedding_dim))(x_embedding)
x_reshape

x_numerical <- layer_input(shape = 2, name = 'numerical_input')
x_combined <- layer_concatenate(list(x_reshape, x_numerical))
x_combined

x <- x_combined

h <- layer_dense(x, intermediate_dim) %>% 
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  layer_activity_regularization(l1=l1, l2=l2)

z_mean <-
  layer_dense(h, latent_dim)  %>% 
  layer_activation(zmean_activation) %>%
  layer_batch_normalization() %>%
  layer_activity_regularization(l1=l1, l2=l2)

z_log_var <-
  layer_dense(h, latent_dim) %>% 
  layer_activation(zlogvar_activation) %>%
  layer_batch_normalization() %>%
  layer_activity_regularization(l1=l1, l2=l2)

sampling <- function(arg) {
  z_mean <- arg[, 0:(latent_dim - 1)]
  z_log_var <- arg[, latent_dim:(2 * latent_dim - 1)]
  
  epsilon <- K$random_normal(
    shape = c(batch_size, latent_dim),
    mean = 0.,
    stddev = epsilon_std
  )
  z_mean + K$exp(z_log_var / 2) * epsilon
}

z <-
  layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

decoder_h <- layer_dense(units = intermediate_dim)
decoder_mean <-
  layer_dense(units = original_dim, activation = decoded_mean_activation)
decoder_var <-
  layer_dense(units = original_dim, activation = "relu")

h_decoded <- decoder_h(z) %>% 
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  layer_activity_regularization(l1=l1, l2=l2)

x_decoded_mean <- decoder_mean(h_decoded)
x_decoded_var <- decoder_var(h_decoded)

# end-to-end autoencoder
vae <-
  keras_model(inputs = c(x_categorical, x_numerical),
              output = x_decoded_mean)
# encoder
encoder <-
  keras_model(inputs = c(x_categorical, x_numerical),
              output = z_mean)


# Loss --------------------------------------------------------

mse_loss <- function(target, reconstruction) {
  as.double(original_dim) * loss_mean_squared_error(target, reconstruction)
}

kl_loss <- function(target, reconstruction) {
  -0.5 * K$mean(1 + z_log_var - K$square(z_mean) - K$exp(z_log_var), axis = -1L)
}

vae_loss <- function(target, reconstruction) {
  mse_loss(target, reconstruction) + kl_loss(target, reconstruction)
}

vae %>% compile(optimizer = optimizer_adam(lr = learning_rate),
                loss = vae_loss)
vae %>% summary()

# Model training ----------------------------------------------------------

if (model_weights_exist == FALSE) {
  hist <- vae %>% fit(
    x = list(X_train[, 1, drop = FALSE], X_train[, 2:3]),
    y = X_train,
    shuffle = TRUE,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(
      callback_tensorboard(log_dir = "/tmp"),
      callback_early_stopping(patience = 50)
    ),
    validation_data = list(list(X_train[, 1, drop = FALSE], X_train[, 2:3]), X_train),
    verbose = 1
  )
  vae %>% save_model_weights_hdf5(weights_file)
  plot(hist)
} else {
  vae %>% load_model_weights_hdf5(weights_file)
}

#
print(vae %>% evaluate(
  list(X_test_nonfraud[, 1, drop = FALSE], X_test_nonfraud[, 2:3]),
  X_test_nonfraud,
  batch_size = 1
))
print(vae %>% evaluate(list(X_test_fraud[, 1, drop = FALSE], X_test_fraud[, 2:3]), X_test_fraud, batch_size =
                         1))
print(vae %>% evaluate(list(X_train[, 1, drop = FALSE], X_train[, 2:3]), X_train, batch_size =
                         batch_size))


#
X_train_encoded <-
  predict(encoder, list(X_train[, 1, drop = FALSE], X_train[, 2:3]),
          batch_size = batch_size) %>%
  cbind("train")
X_test_fraud_encoded <-
  predict(encoder, list(X_test_fraud[, 1, drop = FALSE], X_test_fraud[, 2:3]), batch_size = batch_size)  %>% cbind("fraud")
X_test_nonfraud_encoded <-
  predict(encoder,
          list(X_test_nonfraud[, 1, drop = FALSE], X_test_nonfraud[, 2:3]),
          batch_size = batch_size)  %>% cbind("nonfraud")

df <-
  rbind(X_train_encoded,
        X_test_nonfraud_encoded,
        X_test_fraud_encoded)

p <- df %>%
  as_data_frame() %>%
  mutate(V1 = as.numeric(V1), V2 = as.numeric(V2)) %>%
  ggplot(aes(x = V1, y = V2, colour = V3)) + geom_point(size = 2, alpha = 0.3) + theme(aspect.ratio = 1)
print(p)
