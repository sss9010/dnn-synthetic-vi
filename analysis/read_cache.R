suppressPackageStartupMessages(library(tidyverse))
cache <- "C:/Users/Siim Sepp/dnn_synthetic_vi/analysis/DNN_Synthetic_VI_cache/html"

e_tr <- new.env(); lazyLoad(file.path(cache,"run-training_dfbafdbaf5c23f72c1711ed4fefc6a57"), envir=e_tr)
e_ls <- new.env(); lazyLoad(file.path(cache,"run-lasso-sr_74b4688228d344a7566193c6992c8e5f"),  envir=e_ls)
e_lb <- new.env(); lazyLoad(file.path(cache,"run-lbfgs_48963494f4eb9021c78773cd30a83bf4"),     envir=e_lb)

cat("=== LOEO mean Pearson r per held-out environment ===\n")
loeo <- e_lb$cv_loeo_lbfgs %>%
  filter(!is.na(r)) %>% group_by(held_env) %>%
  summarise(LBFGS = round(mean(r),3), .groups="drop") %>%
  left_join(e_ls$cv_loeo_lasso %>% filter(!is.na(r)) %>% group_by(held_env) %>%
              summarise(LASSO = round(mean(r),3), .groups="drop"), by="held_env") %>%
  left_join(e_tr$cv_loeo %>% filter(!is.na(r)) %>% group_by(held_env) %>%
              summarise(DNN = round(mean(r),3), .groups="drop"), by="held_env")
print(as.data.frame(loeo))
cat(sprintf("Mean r:  L-BFGS=%.3f  LASSO=%.3f  DNN=%.3f\n\n",
    mean(loeo$LBFGS,na.rm=TRUE), mean(loeo$LASSO,na.rm=TRUE), mean(loeo$DNN,na.rm=TRUE)))

cat("=== Within-env 5-fold CV mean Pearson r per environment ===\n")
wenv <- e_lb$cv_wenv_lbfgs %>%
  filter(!is.na(r)) %>% group_by(Env) %>%
  summarise(LBFGS = round(mean(r),3), .groups="drop") %>%
  left_join(e_ls$cv_wenv_lasso %>% filter(!is.na(r)) %>% group_by(Env) %>%
              summarise(LASSO = round(mean(r),3), .groups="drop"), by="Env") %>%
  left_join(e_tr$cv_wenv %>% filter(!is.na(r)) %>% group_by(Env) %>%
              summarise(DNN = round(mean(r),3), .groups="drop"), by="Env")
print(as.data.frame(wenv))
cat(sprintf("Mean r:  L-BFGS=%.3f  LASSO=%.3f  DNN=%.3f\n",
    mean(wenv$LBFGS,na.rm=TRUE), mean(wenv$LASSO,na.rm=TRUE), mean(wenv$DNN,na.rm=TRUE)))
