###################### Lessons learned ##########################
# variables should be of similar scale
# loss function must match the data
# mix of one-hot-encoded and continuous data does not work well


#source("data_UCSD.R")
#source("data_mnist.R")
#source("data_fraud.R")
source("data_unsw.R")


model_weights_exist <- FALSE
weights_file <- "weights_unsw_rsme_xepochs_latent3.h5"


library(keras)

(K <- keras::backend())


# Dataset-dependent parameters --------------------------------------------------------------

# change this according to dataset
# original_dim <- 37604L  # UCSD
# original_dim <- 728L    # MNIST
# original_dim <- 58L     # fraud  
original_dim <- 188L      # attack


# change this too
#latent_dim <- 328L       # UCSD
#latent_dim <- 2L         # MNIST
#latent_dim <- 3L         # fraud
latent_dim <- 2L          # attack


# and this
# intermediate_dim <- 1190L  # UCSD
# intermediate_dim <- 256L   # MNIST
# intermediate_dim <- 8L     # fraud
intermediate_dim <- 32L      # attack


# Tuning parameters --------------------------------------------------------------

batch_size <- 100L
epochs <- 10L
epsilon_std <- 1.0
# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb
#var_epsilon <- 0.025
var_epsilon <- 0.1

# Model definition --------------------------------------------------------

x <- layer_input(shape = original_dim)
h <- layer_dense(x, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_mean
z_log_var <- layer_dense(h, latent_dim)

# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb
#z_mean <- layer_dense(h, latent_dim, activation = "relu")
#z_log_var <- layer_dense(h, latent_dim, activation = "relu")
# add dropout???


sampling <- function(arg){
  z_mean <- arg[,0:(latent_dim-1)]
  z_log_var <- arg[,latent_dim:(2*latent_dim-1)]
  
  epsilon <- K$random_normal(
    shape = c(batch_size, latent_dim), 
    mean=0.,
    stddev=epsilon_std
  )
  #z_mean # deterministic
  z_mean + K$exp(z_log_var/2)*epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)
z

decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
decoder_var <- layer_dense(units = original_dim, activation = "relu")

h_decoded <- decoder_h(z)
x_decoded_mean <- decoder_mean(h_decoded)
x_decoded_var <- decoder_var(h_decoded)


# end-to-end autoencoder
vae <- keras_model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_h(decoder_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2)

# Loss --------------------------------------------------------

xent_loss <- function(target, reconstruction) {
  # multiply by number of rows because Keras returns mean crossentropy, not sum
  as.double(original_dim) * loss_binary_crossentropy(target, reconstruction)
  
}

mse_loss <- function(target, reconstruction) {
  as.double(original_dim) * loss_mean_squared_error(target, reconstruction)
  #loss_mean_squared_error(target, reconstruction)
}



# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb
# https://www.reddit.com/r/MachineLearning/comments/4eqifs/gaussian_observation_vae/

### see for an explanation!
# http://bjlkeng.github.io/posts/a-variational-autoencoder-on-the-svnh-dataset/

logx_loss <- function(target, reconstruction) {
  loss <- 0.5 * K$sum(log(2 * pi) + 
                       K$log(x_decoded_var + var_epsilon) +
                       K$square(x - x_decoded_mean) / (x_decoded_var + var_epsilon), axis = -1L)
}

kl_loss <- function(target, reconstruction) {
  -0.5*K$mean(1 + z_log_var - K$square(z_mean) - K$exp(z_log_var), axis = -1L)
}

vae_loss <- function(target, reconstruction) {
  # optimizing this ends up the same as optimizing the average, i.e
  # K$mean(xent_loss(target, reconstruction) + kl_loss(target, reconstruction))
  #xent_loss(target, reconstruction) + kl_loss(target, reconstruction)
  logx_loss(target, reconstruction) + kl_loss(target, reconstruction)
}

vae %>% compile(optimizer = optimizer_adam(lr=0.0001), loss = vae_loss, metrics = c(mse_loss, logx_loss, kl_loss))
#vae %>% compile(optimizer = optimizer_rmsprop(), loss = vae_loss)


# Model training ----------------------------------------------------------

if (model_weights_exist == FALSE) {
  vae %>% fit(
    X_train, X_train, 
    shuffle = TRUE, 
    epochs = epochs, 
    batch_size = batch_size,
    #validation_data = list(x_test, x_test),
    verbose=1
  ) 
  vae %>% save_model_weights_hdf5(weights_file)
} else {
  vae %>% load_model_weights_hdf5(weights_file)
}


# # Visualize ----------------------------------------------------------------------------
# 
# source("visualize_mnist.R")
# 
# 

# ----------------------------------------------------------------------------------------

first1 <- X_train[1, ]
first1 <- t(first1)

# evaluate loss
vae %>% evaluate(X_train, X_train, batch_size=batch_size)


# View reconstruction / predictions ----------------------------------------------------------

preds <- vae %>% predict(X_train[1:100,], batch_size=batch_size)
dim(preds)
preds


# reconstruction error on test set

vae %>% evaluate(X_test, X_test, batch_size=batch_size)

#vae %>% evaluate(X_test_nonfraud, X_test_nonfraud, batch_size=batch_size)
#vae %>% evaluate(X_test_fraud, X_test_fraud, batch_size=batch_size)

vae %>% evaluate(X_test_nonfraud, X_test_nonfraud, batch_size=batch_size)
vae %>% evaluate(X_test_fraud, X_test_fraud, batch_size=batch_size)
