library(keras)
(K <- keras::backend())

# Data preparation --------------------------------------------------------

#http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm

# https://github.com/y0ast/Variational-Autoencoder/issues/5

"Unfortunately the VAE does not work particularly well on data where n << d, such as neuroimaging data.
There is ongoing research to combat this, but for now I would recommend trying other methods.
As for the reason of the NaN, it's probably the KL divergence blowing up, because the variance is huge.
It's mainly caused by the fact that inference (finding q(z|x)) is probably very hard for your problem.
This results in large variances, which in turn results in a large KL divergence.
You could try with a wider prior on q(z) (essentially softening regularizing grip of the KLD), see also: http://arxiv.org/abs/1511.05644"

library(EBImage)
img_path <- "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/001.tif"
img <- readImage(img_path)
img
dim(img) #238 158

img_matrix <- img@.Data
dim(img_matrix)
img_vector <- as.vector(t(img_matrix))
length(img_vector)


train_dir <- "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
img_files <- list.files(train_dir, recursive = TRUE, full.names = TRUE)
X_train <- matrix(nrow = length(img_files), ncol = 37604)
dim(X_train)

for(i in seq_along(img_files)) {
  img <- readImage(img_files[i])
  img_matrix <- img@.Data
  img_vector <- as.vector(t(img_matrix))
  X_train[i, ] <- img_vector
}


# Parameters --------------------------------------------------------------

batch_size <- 100L
original_dim <- 37604L
latent_dim <- 2L
intermediate_dim <- 1190L
epochs <- 100L
epsilon_std <- 1.0


# Model definition --------------------------------------------------------

x <- layer_input(shape = original_dim)
h <- layer_dense(x, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_mean
z_log_var <- layer_dense(h, latent_dim)

sampling <- function(arg){
  z_mean <- arg[,0:1]
  z_log_var <- arg[,2:3]
  
  epsilon <- K$random_normal(
    shape = c(batch_size, latent_dim), 
    mean=0.,
    stddev=epsilon_std
  )
  #z_mean
  z_mean + K$exp(z_log_var/2)*epsilon
}

# note that "output_shape" isn't necessary with the TensorFlow backend
z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)
z

# we instantiate these layers separately so as to reuse them later
decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
h_decoded <- decoder_h(z)
x_decoded_mean <- decoder_mean(h_decoded)

# end-to-end autoencoder
vae <- keras_model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_h(decoder_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2)


vae_loss <- function(x, x_decoded_mean){
  # https://www.reddit.com/r/MachineLearning/comments/62l7ur/d_binary_crossentropy_as_reconstruction_loss_in/
  #xent_loss <- (original_dim/1.0) * loss_mean_squared_error(x, x_decoded_mean)
  xent_loss <- (original_dim/1.0) * loss_binary_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*K$mean(1 + z_log_var - K$square(z_mean) - K$exp(z_log_var), axis = -1L)
  #kl_loss <- loss_kullback_leibler_divergence(x, x_decoded_mean)
  xent_loss + kl_loss
}

vae %>% compile(optimizer = optimizer_adam(lr=0.0001), loss = vae_loss)
#vae %>% compile(optimizer = optimizer_rmsprop(), loss = vae_loss)



# Model training ----------------------------------------------------------

vae %>% fit(
  X_train, X_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size,
  #validation_data = list(x_test, x_test),
  verbose=1
)

vae %>% save_model_hdf5("vae_1190_2_xent.h5")

# Inspect intermediate layers ----------------------------------------------------------

#layer_name = 'my_layer'
#intermediate_layer_model = Model(inputs=model.input,
#                                 outputs=model.get_layer(layer_name).output)
#intermediate_output = intermediate_layer_model.predict(data)


first100 <- X_train[1:100,]
#dim(first) <- c(1,37604)
dim(first100)
lat <- encoder %>% predict(first100, batch_size=100)
preds <- vae %>% predict(first100, batch_size=100)
dim(preds)

lat[1:5,]
preds[1:5,1:8]

first100[1:5,1:8]
