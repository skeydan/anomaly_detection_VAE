### general ###

model_weights_exist <- FALSE
#weights_file <- "weights_unsw_mse_1000epochs_latent2.h5"
weights_file <- "weights_unsw_mse_new_500epochs_latent2.h5"

### dimensions ###
 
original_dim <- 194L    
#original_dim <- 39L
latent_dim <- 2L         
intermediate_dim <- 32L 
#intermediate_dim <- 8L  

### hyperparameters ###
l1=0
l2=0.1
batch_size <- 100L
epochs <- 500L
learning_rate <- 0.0001


### model ###
zmean_activation="linear"
zlogvar_activation="relu"
decoded_mean_activation="linear"
loss <- "mse"
