# X_test_backdoors is empty!
losses <- Map(function(x) vae %>% evaluate(x,x, batch_size=1),
              list(X_test_normal, X_test_analysis, X_test_DoS,
                   X_test_exploits, X_test_fuzzers, X_test_generic,
                   X_test_reconnaissance, X_test_shellcode, X_test_worms)) 

#losses %>% unlist(Map(function(x) x[[1]], losses))
for (i in seq_along(losses)) {
  print(losses[[i]][[1]])
}

# [1] 37.72816
# [1] 75.22242
# [1] 51.81127
# [1] 47.71854
# [1] 31.7678
# [1] 37.93029
# [1] 18.14437
# [1] 17.03525
# [1] 34.46102