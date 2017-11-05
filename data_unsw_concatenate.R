library(dplyr)
library(ggplot2)
library(readr)


normalize <- function(x){
  (x - min(x))/(max(x)-min(x))
}

train_file <- "UNSW_NB15_training-set.csv"
test_file <- "UNSW_NB15_testing-set.csv"

df_train <- read_csv(train_file)
df_test <- read_csv(test_file)


### reorder columns

df_train <- df_train %>% select(3,4,5,1,2,6:45)
df_test <- df_test %>% select(3,4,5,1,2,6:45)

##########

df_train <- df_train %>% mutate(proto = factor(proto),
                                service = factor(service),
                                state = factor(state),
                                attack_cat = factor(attack_cat))
df_test <- df_test %>% mutate(proto = factor(proto),
                              service = factor(service),
                              state = factor(state),
                              attack_cat = factor(attack_cat))

levels(df_train$proto)
levels(df_train$proto) <- c(levels(df_train$proto), levels(df_test$proto))
levels(df_train$proto)

levels(df_train$state)
levels(df_train$state) <- c(levels(df_train$state), levels(df_test$state))
levels(df_train$state)

levels(df_test$state)
levels(df_test$state) <- c(levels(df_train$state), levels(df_test$state))
levels(df_test$state)

#df_train <- df_train %>% mutate_if(is.numeric, normalize)
df_train <- df_train %>% mutate_if(is.numeric, scale)
summary(df_train)
dim(df_train)

#df_test <- df_test %>% mutate_if(is.numeric, normalize)
df_test <- df_test %>% mutate_if(is.numeric, scale)
summary(df_test)
dim(df_test)

### factor -> numeric

df_train <- df_train %>% mutate(proto = as.numeric(proto),
                                service = as.numeric(service),
                                state = as.numeric(state))
df_test <- df_test %>% mutate(proto = as.numeric(proto),
                              service = as.numeric(service),
                              state = as.numeric(state))

str(df_train)
str(df_test)

###############################################
# training set: normal only
###############################################

table(df_train$label, df_train$attack_cat)

df_train <- df_train %>% filter(attack_cat == "Normal") %>%
                         select(-c(label, attack_cat))
colnames(df_train)

X_train <- as.matrix(df_train)
dim(X_train)


###############################################
# test set
###############################################

table(df_test$label, df_test$attack_cat)

X_test <- as.matrix(df_test %>% select(-c(attack_cat, label)))
y_test <- as.matrix(df_test %>% select(c(attack_cat, label)) %>% mutate_all(as.numeric))
table(y_test)

X_test_normal <- as.matrix(df_test %>% filter(attack_cat == "Normal") %>% select(-c(attack_cat, label)))
X_test_analysis <- as.matrix(df_test %>% filter(attack_cat == "Analysis") %>% select(-c(attack_cat, label)))
X_test_fuzzers <- as.matrix(df_test %>% filter(attack_cat == "Fuzzers") %>% select(-c(attack_cat, label)))
X_test_backdoors <- as.matrix(df_test %>% filter(attack_cat == "Backdoors") %>% select(-c(attack_cat, label)))
X_test_DoS <- as.matrix(df_test %>% filter(attack_cat == "DoS") %>% select(-c(attack_cat, label)))
X_test_exploits <- as.matrix(df_test %>% filter(attack_cat == "Exploits") %>% select(-c(attack_cat, label)))
X_test_generic <- as.matrix(df_test %>% filter(attack_cat == "Generic") %>% select(-c(attack_cat, label)))
X_test_reconnaissance <- as.matrix(df_test %>% filter(attack_cat == "Reconnaissance") %>% select(-c(attack_cat, label)))
X_test_shellcode <- as.matrix(df_test %>% filter(attack_cat == "Shellcode") %>% select(-c(attack_cat, label)))
X_test_worms <- as.matrix(df_test %>% filter(attack_cat == "Worms") %>% select(-c(attack_cat, label)))

Map(function(x) dim(x)[1], list(X_test, X_test_analysis, X_test_backdoors, X_test_DoS,
              X_test_exploits, X_test_fuzzers, X_test_generic, X_test_normal,
              X_test_reconnaissance, X_test_shellcode, X_test_worms)) %>% unlist()


