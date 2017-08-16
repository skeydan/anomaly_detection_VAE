### general ###

model_weights_exist <- FALSE
weights_file <- "weights_ucsd_xent_1000epochs_latent2.h5"

### dimensions ###

original_dim <- 37604L  
latent_dim <- 2L       
intermediate_dim <- 1190L  


### hyperparameters ###

batch_size <- 100L
epochs <- 100L
learning_rate <- 0.0001

### model ###

loss <- "xent"