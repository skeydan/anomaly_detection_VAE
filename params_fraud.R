### general ###

model_weights_exist <- FALSE
weights_file <- "weights_fraud_xent_20epochs_bins10_latent2.h5"


### dimensions ###

#original_dim <- 58L  
original_dim <- 118L     
latent_dim <- 2L        
intermediate_dim <- 32L     


### hyperparameters ###

batch_size <- 100L
epochs <- 1000L
learning_rate <- 0.0001

### model ###

loss <- "xent"
epsilon_std <- 2.0