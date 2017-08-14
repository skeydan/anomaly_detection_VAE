library(dplyr)
library(ggplot2)

options(tibble.print_max = 200, tibble.print_min = 200)
options(scipen = 10)

data(sales, package="DMwR2")

sales <- filter(sales,!(is.na(Quant)))
sales <- filter(sales,!(is.na(Val)))

# this shows that we should NOT reduce things to Uprice because fraud may also be high quantity with low uprice!!!
#sales_1125 <- sales %>% filter(Prod=="p1125", Insp %in% c("ok", "fraud")) %>% arrange(desc(Insp))

sales <- droplevels(sales %>% filter(Insp %in% c("ok", "fraud")))

sales <- sales %>% select(-c(ID))

sales <- sales %>% mutate(Insp = as.numeric(Insp)-1)
sales

###############################################
# zoom in on most common products
###############################################

top_20 <- sales %>% group_by(Prod) %>% summarise(cnt = n()) %>% arrange(desc(cnt)) %>% head(20) %>% select(Prod) %>% pull()

sales <- droplevels(sales %>% filter(Prod %in% top_20))

nrow(sales)
str(sales)

summary_stats <- sales %>% group_by(Prod, Insp) %>% summarise_all(funs(mean, median, min, max))


###############################################
# bin quantities and values
###############################################

hist(sales$Quant, 20, plot = FALSE)
hist(sales$Val, 20, plot = FALSE)

sales <- sales %>% mutate(Quant_bin = cut(Quant, 20), Val_bin = cut(Val,20))

sales <- sales %>% select(-c(Quant, Val))

###############################################
# one-hot encode products, quantities and values
###############################################

sales <- with(sales,
              data.frame(model.matrix(~ Prod + Quant_bin + Val_bin -1,sales),
                         Insp))
str(sales)


###############################################
# split in test and training sets
###############################################

fraud_indices <- which(sales$Insp == 1)
length(fraud_indices)

non_fraud_indices <- setdiff(1:nrow(sales), fraud_indices)
length(non_fraud_indices)

non_fraud_sample <- sample(non_fraud_indices, length(fraud_indices))
length(non_fraud_sample)

test_samples <- c(non_fraud_sample, fraud_indices)
test_matrix <- as.matrix(sales[test_samples, ])
dim(test_matrix)

train_indices <- setdiff(1:nrow(sales), test_samples)
train_matrix <- as.matrix(sales[train_indices, ])
dim(train_matrix)

X_train <- train_matrix[ ,-(ncol(train_matrix))]
dim(X_train)

X_test <- test_matrix[ ,-(ncol(test_matrix))]
dim(X_test)
X_test

###############################################
# for evaluation
###############################################

X_test_nonfraud <- X_test[1:(nrow(X_test)/2), ]
X_test_fraud <- X_test[((nrow(X_test)/2)+1):(nrow(X_test)), ]


