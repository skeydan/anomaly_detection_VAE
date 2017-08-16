#### Visualizations ####

library(ggplot2)
library(dplyr)


X_test_normal_encoded <- predict(encoder, X_test_normal, batch_size = batch_size)  %>% cbind("normal")
X_test_analysis_encoded <- predict(encoder, X_test_analysis, batch_size = batch_size)  %>% cbind("analysis")
X_test_DoS_encoded <- predict(encoder, X_test_DoS, batch_size = batch_size)  %>% cbind("DoS")
X_test_exploits_encoded <- predict(encoder, X_test_exploits, batch_size = batch_size)  %>% cbind("exploits")
X_test_fuzzers_encoded <- predict(encoder, X_test_fuzzers, batch_size = batch_size)  %>% cbind("fuzzers")
X_test_generic_encoded <- predict(encoder, X_test_generic, batch_size = batch_size)  %>% cbind("generic")
X_test_reconnaissance_encoded <- predict(encoder, X_test_reconnaissance, batch_size = batch_size)  %>% cbind("reconnaissance")
X_test_shellcode_encoded <- predict(encoder, X_test_shellcode, batch_size = batch_size)  %>% cbind("shellcode")
X_test_worms_encoded <- predict(encoder, X_test_worms, batch_size = batch_size)  %>% cbind("worms")


df <- rbind(X_test_normal_encoded, X_test_analysis_encoded, X_test_DoS_encoded,
            X_test_exploits_encoded, X_test_fuzzers_encoded, X_test_generic_encoded,
            X_test_reconnaissance_encoded, X_test_shellcode_encoded, X_test_worms_encoded)

df %>%
  as_data_frame() %>%
  mutate(V1 = as.numeric(V1), V2 = as.numeric(V2)) %>%
  ggplot(aes(x = V1, y = V2, colour = V3)) + geom_point()
