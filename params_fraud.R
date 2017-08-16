### general ###

model_weights_exist <- TRUE
weights_file <- "weights_fraud_mse_100epochs_dims102_32_2.h5"


### dimensions ###

#original_dim <- 58L  
#original_dim <- 118L     
original_dim <- 102L
latent_dim <- 2L        
intermediate_dim <- 32L     


### hyperparameters ###

batch_size <- 100L
epochs <- 100L
learning_rate <- 0.0001

### model ###

loss <- "mse"
