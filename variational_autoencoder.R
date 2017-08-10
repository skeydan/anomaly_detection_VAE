###################### Lessons learned ##########################
# variables should be of similar scale
# loss function must match the data
# mix of one-hot-encoded and continuous data does not work well


#source("data_UCSD.R")
#source("data_mnist.R")
source("data_fraud.R")


model_weights_exist <- FALSE
weights_file <- "weights_fraud_xent_20p.h5"


library(keras)

(K <- keras::backend())


# Dataset-dependent parameters --------------------------------------------------------------

# change this according to dataset
# original_dim <- 37604L  # UCSD
# original_dim <- 728L    # MNIST
original_dim <- 21L    # fraud  


# change this too
#latent_dim <- 328L       # UCSD
#latent_dim <- 2L         # MNIST
latent_dim <- 2L         # fraud


# and this
# intermediate_dim <- 1190L  # UCSD
# intermediate_dim <- 256L   # MNIST
intermediate_dim <- 8L  # fraud


# Tuning parameters --------------------------------------------------------------

batch_size <- 1L
epochs <- 2000L
epsilon_std <- 1.0
# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb
#var_epsilon <- 0.025

# Model definition --------------------------------------------------------

x <- layer_input(shape = original_dim)
h <- layer_dense(x, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_mean
z_log_var <- layer_dense(h, latent_dim)

# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb
#z_mean <- layer_dense(h, latent_dim, activation = "relu")
#z_log_var <- layer_dense(h, latent_dim, activation = "relu")

#### add dropout???


sampling <- function(arg){
  #z_mean <- arg[,0:1]
  #z_log_var <- arg[,2:3]
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

decoder_intermediate <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_final <- layer_dense(units = original_dim, activation = "sigmoid")
decoded_intermediate <- decoder_intermediate(z)
decoded_final <- decoder_final(decoded_intermediate)
decoded_final

# end-to-end autoencoder
vae <- keras_model(x, decoded_final)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
generator_decoder_input <- layer_input(shape = latent_dim)
generator_decoded_intermediate <- decoder_intermediate(generator_decoder_input)
generator_decoded_final <- decoder_final(generator_decoded_intermediate)
generator <- keras_model(generator_decoder_input, generator_decoded_final)

# Loss --------------------------------------------------------

xent_loss <- function(target, reconstruction) {
  # no need to take the mean here, as the result is already 1d -> no difference
  # K$mean(as.double(original_dim) * loss_binary_crossentropy(target, reconstruction))
  # multiply by number of rows because Keras returns mean crossentropy, not sum
  as.double(original_dim) * loss_binary_crossentropy(target, reconstruction)
  
}

mse_loss <- function(target, reconstruction) {
  as.double(original_dim) * loss_mean_squared_error(target, reconstruction)
  #loss_mean_squared_error(target, reconstruction)
}



# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb

#logx_loss <- function(x, x_decoded_mean) {
#  loss <- (  0.5 * math.log(2 * math.pi)
#            + 0.5 * K.log(_x_decoded_var + var_epsilon)
#            + 0.5 * K.square(x - x_decoded_mean) / (_x_decoded_var + var_epsilon))
#  loss = K.sum(loss, axis=-1)
#  K.mean(loss)
#}

kl_loss <- function(target, reconstruction) {
  -0.5*K$mean(1 + z_log_var - K$square(z_mean) - K$exp(z_log_var), axis = -1L)
}

vae_loss <- function(target, reconstruction) {
  # optimizing this ends up the same as optimizing the average, i.e
  # K$mean(xent_loss(target, reconstruction) + kl_loss(target, reconstruction))
  #xent_loss(target, reconstruction) + kl_loss(target, reconstruction)
  # xent_loss(target, reconstruction) 
  xent_loss(target, reconstruction) + kl_loss(target, reconstruction)
}

vae %>% compile(optimizer = optimizer_adam(lr=0.0001), loss = vae_loss, metrics = c(xent_loss, kl_loss))
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
first10 <- X_train[1:10,]


# evaluate loss
vae %>% evaluate(first1, first1, batch_size=batch_size)
vae %>% evaluate(first10, first10, batch_size=batch_size)
vae %>% evaluate(X_train, X_train, batch_size=batch_size)


# View reconstruction / predictions ----------------------------------------------------------

preds <- vae %>% predict(first10, batch_size=batch_size)
dim(preds)
preds


# reconstruction error on test set

non_fraud_half <- X_test[1:(nrow(X_test)/2), ]
fraud_half <- X_test[((nrow(X_test)/2)+1):(nrow(X_test)), ]

vae %>% evaluate(non_fraud_half, non_fraud_half, batch_size=batch_size)
vae %>% evaluate(fraud_half, fraud_half, batch_size=batch_size)
