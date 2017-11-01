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
# normalize quantities and values
###############################################

sales <- sales %>% mutate(Prod = as.numeric(Prod))

sales[1:10, ]
summary(sales$Prod)

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

