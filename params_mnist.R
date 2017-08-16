### general ###

model_weights_exist <- FALSE
weights_file <- "weights_mnist_xent_50epochs_latent2.h5"


### dimensions ###

original_dim <- 728L    
latent_dim <- 2L        
intermediate_dim <- 256L   


### hyperparameters ###

batch_size <- 100L
epochs <- 50L
learning_rate <- 0.0001

### model ###

loss <- "xent"