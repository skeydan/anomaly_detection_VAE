library(ggplot2)
library(dplyr)
library(keras)
(K <- keras::backend())

set.seed(777)

source("data_unsw_concatenate.R")

### general ###
model_weights_exist <- FALSE
weights_file <- "weights_unsw_concatenate_embeddings3_intermediate6.h5"

### dimensions ### 
original_dim <- 43L
latent_dim <- 2L        
intermediate_dim <- 6L 

num_categories <- 133 + 13 + 11
num_numeric <- 40
embedding_dim <- 3
#input_length <- 3

### hyperparameters ###
l1 <- 0
l2 <- 0
batch_size <- 100L
epochs <- 100L
learning_rate <- 0.001

### model ###
zmean_activation <- "linear"
zlogvar_activation <- "relu"
decoded_mean_activation <- "linear"

epsilon_std <- 1.0

### model
x_categorical <- layer_input(shape = 3, name = "categorical_input")
x_embedding <- layer_embedding(input_dim = num_categories,
                               output_dim = embedding_dim#,
                               #input_length = input_length
                               )(x_categorical)
x_embedding
x_reshape <-
  layer_reshape(target_shape = 3 * embedding_dim,
                input_shape = c(3, embedding_dim))(x_embedding)
x_reshape

x_numerical <- layer_input(shape = num_numeric, name = 'numerical_input')
x_combined <- layer_concatenate(list(x_reshape, x_numerical))
x_combined

x <- x_combined

h <- layer_dense(x, intermediate_dim) %>% layer_activation("relu")
z_mean <-
  layer_dense(h, latent_dim)  %>% layer_activation(zmean_activation)
z_log_var <-
  layer_dense(h, latent_dim) %>% layer_activation(zlogvar_activation)

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

h_decoded <- decoder_h(z) %>% layer_activation("relu")
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
    x = list(X_train[, 1:3], X_train[, 4:43]),
    y = X_train,
    shuffle = TRUE,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(
      callback_tensorboard(log_dir = "/tmp"),
      callback_early_stopping(patience = 50)
    ),
    validation_data = list(list(X_train[, 1:3], X_train[, 4:43]), X_train),
    verbose = 1
  )
  vae %>% save_model_weights_hdf5(weights_file)
  plot(hist)
} else {
  vae %>% load_model_weights_hdf5(weights_file)
}


losses <- Map(function(x) vae %>% evaluate(list(x[, 1:3], x[, 4:43]), x, batch_size=1),
              list(X_test_normal, X_test_analysis, X_test_DoS,
                   X_test_exploits, X_test_fuzzers, X_test_generic,
                   X_test_reconnaissance, X_test_shellcode, X_test_worms)) 


for (i in seq_along(losses)) {
  print(losses[[i]][[1]])
}

X_test_normal_encoded <- predict(encoder, list(X_test_normal[, 1:3], X_test_normal[, 4:43]), batch_size = batch_size)  %>% cbind("normal")
X_test_analysis_encoded <- predict(encoder, list(X_test_analysis[, 1:3], X_test_analysis[, 4:43]), batch_size = batch_size)  %>% cbind("analysis")
X_test_DoS_encoded <- predict(encoder, list(X_test_DoS[, 1:3], X_test_DoS[, 4:43]), batch_size = batch_size)  %>% cbind("DoS")
X_test_exploits_encoded <- predict(encoder, list(X_test_exploits[, 1:3], X_test_exploits[, 4:43]), batch_size = batch_size)  %>% cbind("exploits")
X_test_fuzzers_encoded <- predict(encoder, list(X_test_fuzzers[, 1:3], X_test_fuzzers[, 4:43]), batch_size = batch_size)  %>% cbind("fuzzers")
X_test_generic_encoded <- predict(encoder, list(X_test_generic[, 1:3], X_test_generic[, 4:43]), batch_size = batch_size)  %>% cbind("generic")
X_test_reconnaissance_encoded <- predict(encoder, list(X_test_reconnaissance[, 1:3], X_test_reconnaissance[, 4:43]), batch_size = batch_size)  %>% cbind("reconnaissance")
X_test_shellcode_encoded <- predict(encoder, list(X_test_shellcode[, 1:3], X_test_shellcode[, 4:43]), batch_size = batch_size)  %>% cbind("shellcode")
X_test_worms_encoded <- predict(encoder, list(X_test_worms[, 1:3], X_test_worms[, 4:43]), batch_size = batch_size)  %>% cbind("worms")


df <- rbind(X_test_normal_encoded, X_test_analysis_encoded, X_test_DoS_encoded,
            X_test_exploits_encoded, X_test_fuzzers_encoded, X_test_generic_encoded,
            X_test_reconnaissance_encoded, X_test_shellcode_encoded, X_test_worms_encoded)

df %>%
  as_data_frame() %>%
  mutate(V1 = as.numeric(V1), V2 = as.numeric(V2)) %>%
  ggplot(aes(x = V1, y = V2, colour = V3)) + geom_point(alpha=0.6) + 
  theme(aspect.ratio = 1)

