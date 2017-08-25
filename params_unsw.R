### general ###

model_weights_exist <- FALSE
#weights_file <- "weights_unsw_mse_1000epochs_latent2.h5"
weights_file <- "weights_unsw_normal_100epochs_latent2.h5"

### dimensions ###
 
original_dim <- 194L    
#original_dim <- 39L
latent_dim <- 2L         
intermediate_dim <- 32L 
#intermediate_dim <- 8L  


### hyperparameters ###

batch_size <- 100L
epochs <- 100L
learning_rate <- 0.0001

### model ###

loss <- "normal"
