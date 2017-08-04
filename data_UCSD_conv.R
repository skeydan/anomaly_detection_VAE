# From: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm


library(EBImage)
img_path <- "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/001.tif"
img <- readImage(img_path)
img
dim(img) #238 158

img_matrix <- img@.Data
dim(img_matrix)

train_dir <- "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
img_files <- list.files(train_dir, recursive = TRUE, full.names = TRUE)
X_train <- array(dim = c(length(img_files), dim(img_matrix)))
dim(X_train)

for(i in seq_along(img_files)) {
  img <- readImage(img_files[i])
  img_matrix <- img@.Data
  X_train[i, , ] <- img_matrix
}

dim(X_train)

dim(X_train) <- c(dim(X_train), 1)

dim(X_train)

