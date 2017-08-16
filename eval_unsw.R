# X_test_backdoors is empty!
losses <- Map(function(x) vae %>% evaluate(x,x, batch_size=1),
              list(X_test_normal, X_test_analysis, X_test_DoS,
                   X_test_exploits, X_test_fuzzers, X_test_generic,
                   X_test_reconnaissance, X_test_shellcode, X_test_worms)) 

#losses %>% unlist(Map(function(x) x[[1]], losses))
for (i in seq_along(losses)) {
  print(losses[[i]][[1]])
}
# [1] 5.927901
# [1] 5.808738
# [1] 6.064029
# [1] 6.569344
# [1] 7.630632
# [1] 7.477292
# [1] 7.15397
# [1] 7.010989
# [1] 8.970896
