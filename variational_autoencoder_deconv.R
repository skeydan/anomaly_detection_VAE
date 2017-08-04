source("data_UCSD_conv.R")
source("data_mnist_conv.R")

library(keras)
(K <- keras::backend())


# Parameters --------------------------------------------------------------

filters <- 64L
num_conv <- 3L # convolution kernel size
latent_dim <- 2L
intermediate_dim <- 128L
epsilon_std <- 1.0

batch_size <- 100L
epochs <- 1L


#### Model Construction ####

original_img_size <- c(img_rows, img_cols, img_chns)

x <- layer_input(batch_shape = c(batch_size, original_img_size))

conv_1 <- layer_conv_2d(
  x,
  filters = img_chns,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_2 <- layer_conv_2d(
  conv_1,
  filters = filters,
  kernel_size = c(2L, 2L),
  strides = c(2L, 2L),
  padding = "same",
  activation = "relu"
)

conv_3 <- layer_conv_2d(
  conv_2,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_4 <- layer_conv_2d(
  conv_3,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

flat <- layer_flatten(conv_4)
hidden <- layer_dense(flat, units = intermediate_dim, activation = "relu")

z_mean <- layer_dense(hidden, units = latent_dim)
z_log_var <- layer_dense(hidden, units = latent_dim)

sampling <- function(args) {
  z_mean <- args[, 0:(latent_dim - 1)]
  z_log_var <- args[, latent_dim:(2 * latent_dim - 1)]
  
  epsilon <- K$random_normal(
    shape = c(batch_size, latent_dim),
    mean = 0.,
    stddev = epsilon_std
  )
  z_mean + K$exp(z_log_var) * epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

output_shape <- c(batch_size, 14L, 14L, filters)

decoder_hidden <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_upsample <- layer_dense(units = prod(output_shape[-1]), activation = "relu")

decoder_reshape <- layer_reshape(target_shape = output_shape[-1])
decoder_deconv_1 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_2 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_3_upsample <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(3L, 3L),
  strides = c(2L, 2L),
  padding = "valid",
  activation = "relu"
)

decoder_mean_squash <- layer_conv_2d(
  filters = img_chns,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "valid",
  activation = "sigmoid"
)

hidden_decoded <- decoder_hidden(z)
up_decoded <- decoder_upsample(hidden_decoded)
reshape_decoded <- decoder_reshape(up_decoded)
deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
x_decoded_relu <- decoder_deconv_3_upsample(deconv_2_decoded)
x_decoded_mean_squash <- decoder_mean_squash(x_decoded_relu)

# custom loss function
vae_loss <- function(x, x_decoded_mean_squash) {
  x <- K$flatten(x)
  x_decoded_mean_squash <- K$flatten(x_decoded_mean_squash)
  xent_loss <- 1.0 * img_rows * img_cols *
    loss_binary_crossentropy(x, x_decoded_mean_squash)
  kl_loss <- -0.5 * K$mean(1 + z_log_var - K$square(z_mean) -
                             K$exp(z_log_var), axis = -1L)
  K$mean(xent_loss + kl_loss)
}

## variational autoencoder
vae <- keras_model(x, x_decoded_mean_squash)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
summary(vae)

## encoder: model to project inputs on the latent space
encoder <- keras_model(x, z_mean)

## build a digit generator that can sample from the learned distribution
gen_decoder_input <- layer_input(shape = latent_dim)
gen_hidden_decoded <- decoder_hidden(gen_decoder_input)
gen_up_decoded <- decoder_upsample(gen_hidden_decoded)
gen_reshape_decoded <- decoder_reshape(gen_up_decoded)
gen_deconv_1_decoded <- decoder_deconv_1(gen_reshape_decoded)
gen_deconv_2_decoded <- decoder_deconv_2(gen_deconv_1_decoded)
gen_x_decoded_relu <- decoder_deconv_3_upsample(gen_deconv_2_decoded)
gen_x_decoded_mean_squash <- decoder_mean_squash(gen_x_decoded_relu)
generator <- keras_model(gen_decoder_input, gen_x_decoded_mean_squash)


# Model training ----------------------------------------------------------

vae %>% fit(
  X_train, X_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size,
  validation_data = list(X_test, X_test),
  verbose=1
)


# Save model weights ------------------------------------------------------------------

vae %>% save_model_weights_hdf5("vae_conv.h5")

# Visualize ----------------------------------------------------------------------------

source("visualize_mnist.R")


# View reconstruction / predictions ----------------------------------------------------------






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
