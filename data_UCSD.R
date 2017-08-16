# From: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm

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

X_train[1:10,1:10]

###

test_dir <- "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"
img_files <- list.files(test_dir, pattern = "*tif", recursive = TRUE, full.names = TRUE)
X_test <- matrix(nrow = length(img_files), ncol = 37604)
dim(X_test)

for(i in seq_along(img_files)) {
  img <- readImage(img_files[i])
  img_matrix <- img@.Data
  img_vector <- as.vector(t(img_matrix))
  X_test[i, ] <- img_vector
}

X_test[1:10,1:10]

