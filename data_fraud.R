library(dplyr)
library(ggplot2)

data(sales, package="DMwR2")
sales <- filter(sales,!(is.na(Quant) & is.na(Val)))
sales <- filter(sales,!(Prod %in% c("p2442", "p2443"))) %>% droplevels()

sales <- mutate(sales,Uprice=Val/Quant)
tPrice <- filter(sales, Insp != "fraud") %>% 
  group_by(Prod) %>% 
  summarise(medianPrice = median(Uprice,na.rm=TRUE))

noQuantMedPrices <- filter(sales, is.na(Quant)) %>% 
  inner_join(tPrice) %>% 
  select(medianPrice)
noValMedPrices <- filter(sales, is.na(Val)) %>% 
  inner_join(tPrice) %>% 
  select(medianPrice)

noQuant <- which(is.na(sales$Quant))
noVal <- which(is.na(sales$Val))

sales[noQuant,'Quant'] <- ceiling(sales[noQuant,'Val'] /noQuantMedPrices)
sales[noVal,'Val'] <- sales[noVal,'Quant'] * noValMedPrices
sales$Uprice <- sales$Val/sales$Quant

#nrow(sales)
#table(sales$Insp)

sales <- droplevels(sales %>% filter(Insp %in% c("ok", "fraud")))

#table(sales$Insp)
#nrow(sales)

sales <- sales %>% select(-c(ID, Quant, Val))


sales <- sales %>% mutate(Insp = as.numeric(Insp)-1)
sales

###############################################
# zoom in on most common products
###############################################

top_20 <- sales %>% group_by(Prod) %>% summarise(cnt = n()) %>% arrange(desc(cnt)) %>% head(20) %>% select(Prod) %>% pull()

sales <- droplevels(sales %>% filter(Prod %in% top_20))

nrow(sales)
str(sales)
sales


###############################################
# log-transform Uprice
###############################################

sales <- sales %>% mutate(Uprice = log(Uprice))
summary(sales$Uprice)


###############################################
# scale Uprice
###############################################

# if we want to use binary crossentropy Uprice has to be between 0 and 1
# Uprice should be similar in size to the product 0-1 bits also for MSE!

normalize <- function(vec, min, max) {
  (vec-min) / (max-min)
}
minval <- min(sales$Uprice)
maxval <- max(sales$Uprice)
minval
maxval

sales <- sales %>% mutate(Uprice = normalize(Uprice, minval, maxval))
summary(sales$Uprice)



###############################################
# one-hot encode products
###############################################


sales <- with(sales,
            data.frame(model.matrix(~Prod-1,sales),
                       Uprice,Insp))

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
