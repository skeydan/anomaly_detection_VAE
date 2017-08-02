# From: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm

# https://github.com/y0ast/Variational-Autoencoder/issues/5

"Unfortunately the VAE does not work particularly well on data where n << d, such as neuroimaging data.
There is ongoing research to combat this, but for now I would recommend trying other methods.
As for the reason of the NaN, it's probably the KL divergence blowing up, because the variance is huge.
It's mainly caused by the fact that inference (finding q(z|x)) is probably very hard for your problem.
This results in large variances, which in turn results in a large KL http://bjlkeng.github.io/posts/a-variational-autoencoder-on-the-svnh-dataset/divergence.
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

