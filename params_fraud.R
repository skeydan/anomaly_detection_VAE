### general ###
model_weights_exist <- TRUE
weights_file <- "weights_fraud.h5"

### dimensions ### 
original_dim <- 102L
latent_dim <- 2L        
intermediate_dim <- 32L  

### hyperparameters ###
l1 <- 0
l2 <- 0
batch_size <- 100L
epochs <- 500L
learning_rate <- 0.001
use_optimizer <- "adam"

### model ###
zmean_activation <- "linear"
zlogvar_activation <- "relu"
decoded_mean_activation <- "linear"
use_batch_normalization <- FALSE
loss <- "mse"