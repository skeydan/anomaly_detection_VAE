### general ###
model_weights_exist <- TRUE
weights_file <- "weights_mnist_xent_50epochs_latent2.h5"

### dimensions ###
original_dim <- 784L    
latent_dim <- 2L        
intermediate_dim <- 256L 

### hyperparameters ###
l1=0
l2=0
batch_size <- 100L
epochs <- 50L
learning_rate <- 0.001
use_optimizer <- "rmsprop"


### model ###
zmean_activation="linear"
zlogvar_activation="linear"
decoded_mean_activation="sigmoid"
loss <- "xent"
use_batch_normalization <- FALSE
