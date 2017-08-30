#### Visualizations ####

library(ggplot2)
library(dplyr)

X_train_encoded <- predict(encoder, X_train, batch_size = batch_size) %>% cbind("train")
X_test_fraud_encoded <- predict(encoder, X_test_fraud, batch_size = batch_size)  %>% cbind("fraud")
X_test_nonfraud_encoded <- predict(encoder, X_test_nonfraud, batch_size = batch_size)  %>% cbind("nonfraud")

df <- rbind(X_train_encoded, X_test_fraud_encoded, X_test_nonfraud_encoded)

p <- df %>%
  as_data_frame() %>%
  mutate(V1 = as.numeric(V1), V2 = as.numeric(V2)) %>%
  ggplot(aes(x = V1, y = V2, colour = V3)) + geom_point(size=2, alpha = 0.4) + theme(aspect.ratio = 1)
print(p)
