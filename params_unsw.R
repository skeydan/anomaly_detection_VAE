### general ###

model_weights_exist <- TRUE
weights_file <- "weights_unsw_rsme_1000epochs_latent2.h5"


### dimensions ###
 
original_dim <- 194L      
latent_dim <- 2L         
intermediate_dim <- 32L      


### hyperparameters ###

batch_size <- 100L
epochs <- 1000L
learning_rate <- 0.0001

### model ###

loss <- "mse"
