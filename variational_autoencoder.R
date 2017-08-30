library(keras)
library(tensorflow)
(K <- keras::backend())

set.seed(777)

#source("data_UCSD.R")
#source("data_mnist.R")
#source("data_fraud3.R")
source("data_unsw.R")


#source("params_UCSD.R")
#source("params_mnist.R")
#source("params_fraud.R")
source("params_unsw.R")


# Tuning parameters --------------------------------------------------------------

epsilon_std <- 1.0
# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb
var_epsilon <- 0.01

# Model definition --------------------------------------------------------

x <- layer_input(shape = original_dim)
h <- layer_dense(x, intermediate_dim) 
if(use_batch_normalization) h <- h %>% layer_batch_normalization() 
h <- h %>% layer_activation("relu") %>% layer_activity_regularization(l1=l1, l2=l2)

z_mean <- layer_dense(h, latent_dim) 
if(use_batch_normalization) z_mean <- z_mean %>% layer_batch_normalization() 
z_mean <- z_mean %>% layer_activation(zmean_activation) %>% layer_activity_regularization(l1=l1, l2=l2)

z_log_var <- layer_dense(h, latent_dim) 
if(use_batch_normalization) z_log_var <- z_log_var %>% layer_batch_normalization() 
z_log_var <- z_log_var %>% layer_activation(zlogvar_activation) %>% layer_activity_regularization(l1=l1, l2=l2)


sampling <- function(arg){
  z_mean <- arg[,0:(latent_dim-1)]
  z_log_var <- arg[,latent_dim:(2*latent_dim-1)]
  
  epsilon <- K$random_normal(
    shape = c(batch_size, latent_dim), 
    mean=0.,
    stddev=epsilon_std
  )
  z_mean + K$exp(z_log_var/2)*epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling) 

decoder_h <- layer_dense(units = intermediate_dim)
decoder_mean <- layer_dense(units = original_dim, activation = decoded_mean_activation)  
decoder_var <- layer_dense(units = original_dim, activation = "relu") 

h_decoded <- decoder_h(z) 
if(use_batch_normalization) h_decoded <- h_decoded %>% layer_batch_normalization() 
h_decoded <- h_decoded %>% layer_activation("relu") %>% layer_activity_regularization(l1=l1, l2=l2)
x_decoded_mean <- decoder_mean(h_decoded)
x_decoded_var <- decoder_var(h_decoded)


# end-to-end autoencoder
vae <- keras_model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_h(decoder_input)
if(use_batch_normalization) h_decoded_2 <- h_decoded_2 %>% layer_batch_normalization() 
h_decoded_2 <- h_decoded_2 %>% layer_activation("relu") %>% layer_activity_regularization(l1=l1, l2=l2)
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

mae_loss <- function(target, reconstruction) {
  as.double(original_dim) * loss_mean_absolute_error(target, reconstruction)
  #loss_mean_squared_error(target, reconstruction)
}

cat_loss <- function(target, reconstruction) {
  as.double(original_dim) * loss_categorical_crossentropy(target, reconstruction)
}

# https://github.com/bjlkeng/sandbox/blob/master/notebooks/variational_autoencoder-svhn/model_fit.ipynb
# https://www.reddit.com/r/MachineLearning/comments/4eqifs/gaussian_observation_vae/
# see for an explanation: http://bjlkeng.github.io/posts/a-variational-autoencoder-on-the-svnh-dataset/
# normal loglikelihood
normal_loss <- function(target, reconstruction) {
  loss_one_col <- 0.5 * log(2 * pi) + 
              K$log(sqrt(x_decoded_var) + var_epsilon) +
              K$square(x - x_decoded_mean) / (x_decoded_var + var_epsilon)
  loss <- K$sum(loss_one_col, axis = -1L)
  loss
}

kl_loss <- function(target, reconstruction) {
  -0.5*K$mean(1 + z_log_var - K$square(z_mean) - K$exp(z_log_var), axis = -1L)
}

vae_loss <- function(target, reconstruction) {
  # optimizing this ends up the same as optimizing the average, i.e
  # K$mean(xent_loss(target, reconstruction) + kl_loss(target, reconstruction))
  switch(loss,
         "normal" = normal_loss(target, reconstruction),
         "xent" = xent_loss(target, reconstruction),
         "mse" = mse_loss(target, reconstruction),
         "mae" = mae_loss(target, reconstruction),
         "cat" = loss_categorical_crossentropy(target, reconstruction)) +
    kl_loss(target, reconstruction)
}

vae %>% compile(optimizer = if(use_optimizer == "rmsprop") {
                               optimizer_rmsprop(lr = learning_rate) 
                            } else {
                            optimizer_adam(lr=learning_rate)
                            },
                loss = vae_loss,
                metrics = c(xent_loss, mse_loss, normal_loss, kl_loss))
vae %>% summary()

# Model training ----------------------------------------------------------

if (model_weights_exist == FALSE) {
  vae %>% fit(
    X_train, X_train, 
    shuffle = TRUE, 
    epochs = epochs, 
    batch_size = batch_size,
    callbacks = list(callback_tensorboard(log_dir="/tmp"), callback_early_stopping(patience=50)),
    validation_data = list(X_train, X_train),
    verbose=1
  ) 
  vae %>% save_model_weights_hdf5(weights_file)
} else {
  vae %>% load_model_weights_hdf5(weights_file)
}


#source("visualize_fraud.R")
source("visualize_unsw.R")
#source("visualize_mnist.R")
 

# evaluate  ---------------------------------------------------------------------------

#vae %>% evaluate(X_train, X_train, batch_size=batch_size)

#source("eval_UCSD.R")
#source("eval_fraud.R")
source("eval_unsw.R")

# get gradients
weights <- vae$trainable_weights 
weights[[1]]$name
outputTensor <- vae$output 
outputTensor
gradients <- K$gradients(outputTensor , weights)
gradients
# 
sess <-tf$InteractiveSession()
sess$run(tf$global_variables_initializer())
# 
input <- vae$input
input
output <- vae$output
output
evaluated_gradients <- sess$run(gradients,
                                feed_dict = dict(input = X_train[1:100, ], output = X_train[1:100, ]))
evaluated_gradients

# latent variable layers
# encoder %>% predict(X_test[1:100, ])
# 
# # View reconstruction / predictions ----------------------------------------------------------
# X_train_100 <- X_train[1:100,]
# preds_100 <- vae %>% predict(X_train_100, batch_size=100)
#X_train_10
#preds_10


