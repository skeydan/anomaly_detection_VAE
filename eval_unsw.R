# X_test_backdoors is empty!
losses <- Map(function(x) vae %>% evaluate(x,x, batch_size=1),
              list(X_test_normal, X_test_analysis, X_test_DoS,
                   X_test_exploits, X_test_fuzzers, X_test_generic,
                   X_test_reconnaissance, X_test_shellcode, X_test_worms)) 

#losses %>% unlist(Map(function(x) x[[1]], losses))
for (i in seq_along(losses)) {
  print(losses[[i]][[1]])
}

# [1] 23.1978
# [1] 72.77035
# [1] 42.99196
# [1] 540530.5
# [1] 23.20095
# [1] 39.8451
# [1] 15.66849
# [1] 14.21412
# [1] 35.61949