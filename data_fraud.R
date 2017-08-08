library(dplyr)
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

###############################################
#nrow(sales)
#table(sales$Insp)

sales <- droplevels(sales %>% filter(Insp %in% c("ok", "fraud")))
#table(sales$Insp)
#nrow(sales)

sales <- sales %>% select(-ID)
sales <- sales %>% mutate(Insp = as.numeric(Insp)-1)
#sales

sales <- with(sales,
            data.frame(model.matrix(~Prod-1,sales),
                       Uprice,Insp))

#str(sales)

fraud_indices <- which(sales$Insp == 1)
#length(fraud_indices)

non_fraud_indices <- setdiff(1:nrow(sales), fraud_indices)
#length(non_fraud_indices)

non_fraud_sample <- sample(non_fraud_indices, length(fraud_indices))
#length(non_fraud_sample)

test_samples <- c(non_fraud_sample, fraud_indices)
test_matrix <- as.matrix(sales[test_samples, ])
#dim(test_matrix)

train_indices <- setdiff(1:nrow(sales), test_samples)
train_matrix <- as.matrix(sales[train_indices, ])
#dim(train_matrix)

X_train <- train_matrix[ ,1:798]

# if we want to use binary crossentropy!
normalize <- function(vec, min, max) {
  (vec-min) / (max-min)
}
minval <- min(X_train[ ,798])
maxval <- max(X_train[ ,798])
#minval
#maxval

X_train[ , 798] <- normalize(X_train[ ,798], minval, maxval)
#summary(X_train[ ,798])
