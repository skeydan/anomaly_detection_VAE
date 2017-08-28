print(vae %>% evaluate(X_test_nonfraud, X_test_nonfraud, batch_size=1))
print(vae %>% evaluate(X_test_fraud, X_test_fraud, batch_size=1))
print(vae %>% evaluate(X_train, X_train, batch_size=batch_size))
