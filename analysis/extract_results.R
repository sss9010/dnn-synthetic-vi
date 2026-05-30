suppressPackageStartupMessages(library(tidyverse))

cache_dir <- "C:/Users/Siim Sepp/dnn_synthetic_vi/analysis/DNN_Synthetic_VI_cache/html"

e_fulldnn <- new.env()
lazyLoad(file.path(cache_dir, "run-fulldnn-training_2ce43b92826d980f1d4a74ce4914d625"), envir = e_fulldnn)

e_lasso <- new.env()
lazyLoad(file.path(cache_dir, "run-lasso-sr_24bf12c8709d056bb346c0f59ec064d3"), envir = e_lasso)

e_train <- new.env()
lazyLoad(file.path(cache_dir, "run-training_06fda461d78c723b4925dbc40762011b"), envir = e_train)

cat("Vars in fulldnn cache:", paste(ls(e_fulldnn), collapse = ", "), "\n")
cat("Vars in lasso cache:  ", paste(ls(e_lasso),   collapse = ", "), "\n")
cat("Vars in train cache:  ", paste(ls(e_train),   collapse = ", "), "\n\n")

# LOEO
cv_loeo         <- e_train$cv_loeo
cv_loeo_fulldnn <- e_fulldnn$cv_loeo_fulldnn
cv_loeo_lasso   <- e_lasso$cv_loeo_lasso

bot_loeo   <- cv_loeo         %>% filter(!is.na(r)) %>% group_by(held_env) %>% summarise(DNN_bottleneck    = round(mean(r), 3), .groups = "drop")
full_loeo  <- cv_loeo_fulldnn %>% filter(!is.na(r)) %>% group_by(held_env) %>% summarise(DNN_no_bottleneck = round(mean(r), 3), .groups = "drop")
lasso_loeo <- cv_loeo_lasso   %>% filter(!is.na(r)) %>% group_by(held_env) %>% summarise(LASSO_poly_SR     = round(mean(r), 3), .groups = "drop")

cat("=== LOEO Pearson r (mean over 3 reps) ===\n")
loeo_comp <- bot_loeo %>%
  left_join(full_loeo,  by = "held_env") %>%
  left_join(lasso_loeo, by = "held_env")
print(as.data.frame(loeo_comp))
cat(sprintf("\nMean r: DNN_bottleneck=%.3f  DNN_no_bottleneck=%.3f  LASSO_poly_SR=%.3f\n\n",
  mean(loeo_comp$DNN_bottleneck, na.rm = TRUE),
  mean(loeo_comp$DNN_no_bottleneck, na.rm = TRUE),
  mean(loeo_comp$LASSO_poly_SR, na.rm = TRUE)))

# Within-env
cv_wenv         <- e_train$cv_wenv
cv_wenv_fulldnn <- e_fulldnn$cv_wenv_fulldnn
cv_wenv_lasso   <- e_lasso$cv_wenv_lasso

bot_wenv   <- cv_wenv         %>% filter(!is.na(r)) %>% group_by(Env) %>% summarise(DNN_bottleneck    = round(mean(r), 3), .groups = "drop")
full_wenv  <- cv_wenv_fulldnn %>% filter(!is.na(r)) %>% group_by(Env) %>% summarise(DNN_no_bottleneck = round(mean(r), 3), .groups = "drop")
lasso_wenv <- cv_wenv_lasso   %>% filter(!is.na(r)) %>% group_by(Env) %>% summarise(LASSO_poly_SR     = round(mean(r), 3), .groups = "drop")

cat("=== Within-env 5-fold CV Pearson r (mean over folds x reps) ===\n")
wenv_comp <- bot_wenv %>%
  left_join(full_wenv,  by = "Env") %>%
  left_join(lasso_wenv, by = "Env")
print(as.data.frame(wenv_comp))
cat(sprintf("\nMean r: DNN_bottleneck=%.3f  DNN_no_bottleneck=%.3f  LASSO_poly_SR=%.3f\n\n",
  mean(wenv_comp$DNN_bottleneck, na.rm = TRUE),
  mean(wenv_comp$DNN_no_bottleneck, na.rm = TRUE),
  mean(wenv_comp$LASSO_poly_SR, na.rm = TRUE)))

# LASSO formula
if ("n_terms" %in% names(cv_loeo_lasso)) {
  cat("=== LASSO active terms per LOEO fold ===\n")
  print(as.data.frame(cv_loeo_lasso %>% select(held_env, r, n_terms) %>% mutate(r = round(r, 3))))
}
