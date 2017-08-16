library(dplyr)
library(ggplot2)

options(tibble.print_max = 200, tibble.print_min = 200)
options(scipen = 10)

data(sales, package="DMwR2")

sales <- filter(sales,!(is.na(Quant)))
sales <- filter(sales,!(is.na(Val)))

# this shows that we should NOT reduce things to Uprice because fraud may also be high quantity with low uprice!!!
#sales_1125 <- sales %>% filter(Prod=="p1125", Insp %in% c("ok", "fraud")) %>% arrange(desc(Insp))

#sales <- droplevels(sales %>% filter(Insp %in% c("ok", "fraud")))

sales <- sales %>% select(-c(ID))

#sales <- sales %>% mutate(Insp = as.numeric(Insp)-1)
nrow(sales)

###############################################
# zoom in on most common products
###############################################

top_100 <- sales %>% group_by(Prod) %>% summarise(cnt = n()) %>% arrange(desc(cnt)) %>% head(100) %>% select(Prod) %>% pull()

sales <- droplevels(sales %>% filter(Prod %in% top_100))

nrow(sales)
str(sales)

summary_stats <- sales %>% group_by(Prod, Insp) %>% summarise_all(funs(mean, median, min, max))

###############################################
# one-hot encode products
###############################################

sales <- with(sales,
              data.frame(model.matrix(~ Prod-1,sales),
                         Insp, Val, Quant))
str(sales)


###############################################
# normalize quantities and values
###############################################

normalize <- function(x){
  (x - min(x))/(max(x)-min(x))
}
summary(sales$Quant)
summary(sales$Val)

sales <- sales %>% mutate(Quant = normalize(Quant))
sales <- sales %>% mutate(Val = normalize(Val))

dim(sales)

###############################################
# split in test and training sets
###############################################

X_train <- sales %>% filter(Insp == "unkn") %>% 
                     select(-Insp) %>% 
                     as.matrix()
dim(X_train)

X_train <- X_train[1:71700, ]

X_test_nonfraud <- sales %>% filter(Insp == "ok") %>% 
  select(-Insp) %>% 
  as.matrix()
dim(X_test_nonfraud)

X_test_fraud <- sales %>% filter(Insp == "fraud") %>% 
  select(-Insp) %>% 
  as.matrix()
dim(X_test_fraud)

