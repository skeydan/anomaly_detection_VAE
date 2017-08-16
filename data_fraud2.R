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

top_40 <- sales %>% group_by(Prod) %>% summarise(cnt = n()) %>% arrange(desc(cnt)) %>% head(40) %>% select(Prod) %>% pull()

sales <- droplevels(sales %>% filter(Prod %in% top_40))

nrow(sales)
str(sales)

summary_stats <- sales %>% group_by(Prod, Insp) %>% summarise_all(funs(mean, median, min, max))


###############################################
# bin quantities and values
###############################################

hist(sales$Quant, 40, plot = FALSE)
hist(sales$Val, 40, plot = FALSE)

sales <- sales %>% mutate(Quant_bin = cut(Quant, 40), Val_bin = cut(Val,40))

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

X_train <- sales %>% filter(Insp == "unkn") %>% 
                     select(-Insp) %>% 
                     as.matrix()
dim(X_train)

X_train <- X_train[1:39800, ]

X_test_nonfraud <- sales %>% filter(Insp == "ok") %>% 
  select(-Insp) %>% 
  as.matrix()
dim(X_test_nonfraud)

X_test_fraud <- sales %>% filter(Insp == "fraud") %>% 
  select(-Insp) %>% 
  as.matrix()
dim(X_test_fraud)

