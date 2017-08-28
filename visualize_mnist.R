#### Visualizations ####

library(ggplot2)
library(dplyr)

## display a 2D plot of the digit classes in the latent space
x_test_encoded <- predict(encoder, X_test, batch_size = batch_size)
p <- x_test_encoded %>%
  as_data_frame() %>%
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()
print(p)

## display a 2D manifold of the digits
n <- 15  # figure with 15x15 digits
digit_size <- 28

# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(-4, 4, length.out = n)
grid_y <- seq(-4, 4, length.out = n)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = digit_size))
  }
  rows <- cbind(rows, column)
}
p <- rows %>% as.raster() %>% plot()
print(p)
