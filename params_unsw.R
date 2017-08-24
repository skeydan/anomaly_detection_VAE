### general ###

model_weights_exist <- FALSE
#weights_file <- "weights_unsw_mse_1000epochs_latent2.h5"
weights_file <- "weights_unsw_mse_1000epochs_latent2_justnumeric.h5"

### dimensions ###
 
original_dim <- 194L    
#original_dim <- 39L
latent_dim <- 2L         
intermediate_dim <- 32L 
#intermediate_dim <- 8L  


### hyperparameters ###

batch_size <- 100L
epochs <- 1000L
learning_rate <- 0.01

### model ###

loss <- "mse"
