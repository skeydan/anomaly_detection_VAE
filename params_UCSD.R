### general ###
model_weights_exist <- FALSE
weights_file <- "weights_UCSD.h5"

### dimensions ###
original_dim <- 37604L  
latent_dim <- 2L       
intermediate_dim <- 1190L  


### hyperparameters ###
l1=0
l2=0
batch_size <- 100L
epochs <- 100L
learning_rate <- 0.0001
use_optimizer <- "adam"


### model ###
zmean_activation="linear"
zlogvar_activation="linear"
decoded_mean_activation="sigmoid"
loss <- "xent"
use_batch_normalization <- FALSE
