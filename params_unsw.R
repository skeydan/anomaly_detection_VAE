### general ###

model_weights_exist <- TRUE
weights_file <- "weights_unsw.h5"

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
use_optimizer <- "adam"

### model ###
zmean_activation="linear"
zlogvar_activation="relu"
decoded_mean_activation="linear"
loss <- "mse"
use_batch_normalization <- TRUE