library(keras)
(K <- keras::backend())

# Data preparation --------------------------------------------------------

#http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm

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
encoding_dim <- 238L
epochs <- 5L


# Model definition --------------------------------------------------------

x <- layer_input(batch_shape = c(batch_size, original_dim))

encoded <- layer_dense(x, units = encoding_dim, activation = "relu")
#encoded <- layer_dense(units = encoding_dim, activation = "relu", activity_regularizer = regularizer_l1())
decoded <- layer_dense(encoded, units = original_dim, activation = "sigmoid")

# end-to-end autoencoder
autoencoder <- keras_model(x, decoded)
autoencoder %>% summary()

# encoder, from inputs to latent space
encoder <- keras_model(x, encoded)
encoder %>% summary()

# decoder 
encoded_input <- layer_input(batch_shape = c(batch_size, encoding_dim))
decoder_layer <- autoencoder %>% get_layer(index=2)
decoder <- keras_model(encoded_input, decoder_layer(encoded_input))
decoder %>% summary()

autoencoder %>% compile(optimizer = "adadelta", loss = loss_binary_crossentropy)



# Model training ----------------------------------------------------------

autoencoder %>% fit(
  X_train, X_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size,
  #validation_data = list(x_test, x_test),
  verbose=1
)


