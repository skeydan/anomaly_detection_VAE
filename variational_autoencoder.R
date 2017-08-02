source("data_UCSD.R")
source("data_mnist.R")

library(keras)
(K <- keras::backend())


# Parameters --------------------------------------------------------------

batch_size <- 100L
original_dim <- 37604L
latent_dim <- 328L
intermediate_dim <- 1190L
epochs <- 10L
epsilon_std <- 1.0
# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb
var_epsilon <- 0.025

# Model definition --------------------------------------------------------

x <- layer_input(shape = original_dim)
h <- layer_dense(x, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_mean
z_log_var <- layer_dense(h, latent_dim)

# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb
z_mean <- layer_dense(h, latent_dim, activation = "relu")
z_log_var <- layer_dense(h, latent_dim, activation = "relu")

#### add dropout???


sampling <- function(arg){
  #z_mean <- arg[,0:1]
  #z_log_var <- arg[,2:3]
  z_mean <- arg[,0:327]
  z_log_var <- arg[,328:655]
  
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
h_decoded <- decoder_h(z)
x_decoded_mean <- decoder_mean(h_decoded)

# end-to-end autoencoder
vae <- keras_model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_h(decoder_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2)

xent_loss <- function(x, x_decoded_mean) {
  (original_dim/1.0) * loss_binary_crossentropy(x, x_decoded_mean)
}



# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb

#logx_loss <- function(x, x_decoded_mean) {
#  loss <- (  0.5 * math.log(2 * math.pi)
#            + 0.5 * K.log(_x_decoded_var + var_epsilon)
#            + 0.5 * K.square(x - x_decoded_mean) / (_x_decoded_var + var_epsilon))
#  loss = K.sum(loss, axis=-1)
#  K.mean(loss)
#}

kl_loss <- function(x, x_decoded_mean) {
  -0.5*K$mean(1 + z_log_var - K$square(z_mean) - K$exp(z_log_var), axis = -1L)
  #loss_kullback_leibler_divergence(x, x_decoded_mean)
}

vae_loss <- xent_loss + kl_loss

vae %>% compile(optimizer = optimizer_adam(lr=0.0001), loss = vae_loss)
#vae %>% compile(optimizer = optimizer_rmsprop(), loss = vae_loss)



# Model training ----------------------------------------------------------

vae %>% fit(
  X_train, X_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size,
  #validation_data = list(x_test, x_test),
  verbose=1
)

#vae %>% save_model_hdf5("vae_1190_2_xent.h5")
vae %>% save_model_hdf5("vae_1190_2_rsme.h5")

# Inspect intermediate layers ----------------------------------------------------------

#layer_name = 'my_layer'
#intermediate_layer_model = Model(inputs=model.input,
#                                 outputs=model.get_layer(layer_name).output)
#intermediate_output = intermediate_layer_model.predict(data)


first100 <- X_train[1:100,]
#dim(first) <- c(1,37604)
dim(first100)
lat <- encoder %>% predict(first100, batch_size=100)
preds <- vae %>% predict(first100, batch_size=100)
dim(preds)

lat[1:5,]
preds[1:5,1:8]

first100[1:5,1:8]
