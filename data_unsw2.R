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

##########

df_train <- df_train %>% mutate(attack_cat = factor(attack_cat))
df_test <- df_test %>% mutate(attack_cat = factor(attack_cat))

df_train <- df_train %>% select_if(function(column) !is.character(column))
df_test <- df_test %>% select_if(function(column) !is.character(column))


df_train <- df_train %>% mutate_if(is.numeric, scale)
#summary(df_train)
dim(df_train)


df_test <- df_test %>% mutate_if(is.numeric, scale)
#summary(df_test)
dim(df_test)



###############################################
# training set: normal only
###############################################

table(df_train$label, df_train$attack_cat)

df_train <- df_train %>% filter(attack_cat == "Normal") %>%
                         select(-c(id,label, attack_cat))
colnames(df_train)

X_train <- as.matrix(df_train)
dim(X_train)


###############################################
# test set
###############################################

table(df_test$label, df_test$attack_cat)

X_test <- as.matrix(df_test %>% select(-c(id,attack_cat, label)))
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

###############################################
# datasets with labels --- 1 column
# for dl4j
###############################################

df_test <- df_test %>% mutate(attack_cat = relevel(attack_cat, ref="Normal"))
levels(df_test$attack_cat)
#[1] "Normal"         "Analysis"       "Backdoor"      
#[4] "DoS"            "Exploits"       "Fuzzers"       
#[7] "Generic"        "Reconnaissance" "Shellcode"     
#[10] "Worms"         

X_test_with_label <- df_test %>% select(-c(label)) %>% mutate(attack_cat = as.numeric(as.factor(attack_cat))-1)

dim(X_test_with_label)
colnames((X_test_with_label))
table(X_test_with_label$attack_cat)

X_train_with_label <- as.data.frame(X_train) %>% mutate(attack_cat = 0)
dim(X_train_with_label)
table(X_train_with_label$attack_cat)

### datasets with labels --- one-hot

# X_test_with_label <- df_test %>% select(-c(label)) 
# X_test_with_label <- X_test_with_label %>% cbind(with(X_test_with_label, model.matrix(~ attack_cat -1))) %>%
#   select(-attack_cat)
# 
# dim(X_test_with_label)
# colnames((X_test_with_label))
# 
# X_train_with_label <- as.data.frame(X_train) %>% mutate(attack_catAnalysis = 0,
#                                                         attack_catBackdoor = 0,
#                                                         attack_catDoS = 0,
#                                                         attack_catExploits = 0,
#                                                         attack_catFuzzers = 0,
#                                                         attack_catGeneric = 0,
#                                                         attack_catNormal = 1,
#                                                         attack_catReconnaissance = 0,
#                                                         attack_catShellcode = 0,
#                                                         attack_catWorms =0)
# colnames(X_train_with_label)
# summary(X_train_with_label)
# 

### write out to csv

library(readr)
X_test_with_label %>% as.data.frame() %>% write_csv("X_test_with_label.csv")
X_train_with_label %>% as.data.frame() %>% write_csv("X_train_with_label.csv")

