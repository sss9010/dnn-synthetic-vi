suppressPackageStartupMessages({
  library(tidyverse)
  library(glmnet)
  library(torch)
})
if (!torch::torch_is_installed()) torch::install_torch()
set.seed(42); torch_manual_seed(42)

# ── Data ──────────────────────────────────────────────────────────────────
pheno_paths <- c(
  "C:/Users/Siim Sepp/NY_WMBCL/data/WMB_pheno.Rdata",
  "~/Documents/GitHub/NY_WMBCL/data/WMB_pheno.Rdata",
  "data/WMB_pheno.Rdata"
)
pheno_file <- pheno_paths[file.exists(pheno_paths)][1]
load(pheno_file)
loaded_objs <- ls()
df_candidates <- loaded_objs[sapply(loaded_objs, function(x) is.data.frame(get(x)))]
if (length(df_candidates) == 1) pheno_raw <- get(df_candidates) else {
  sizes <- sapply(df_candidates, function(x) nrow(get(x)))
  pheno_raw <- get(df_candidates[which.max(sizes)])
}

BAND_NAMES <- c("blue", "green", "red", "red_edge", "nir")
band_patterns <- c(blue="^blue$", green="^green$", red="^red$",
                   red_edge="red[._]?edge|rededge", nir="^nir$")
band_cols <- sapply(names(band_patterns), function(b) {
  m <- grep(band_patterns[b], names(pheno_raw), ignore.case=TRUE, value=TRUE)
  if (length(m)==0) NA_character_ else m[1]
}, USE.NAMES=TRUE)
band_cols <- band_cols[!is.na(band_cols)]

env_col <- intersect(c("Env","env","environment","trial"), names(pheno_raw))[1]
gid_col <- intersect(c("GID","gid","Genotype","genotype","EntryName","entry_name",
                       "Line","line","Name","name"), names(pheno_raw))[1]
tp_col  <- intersect(c("Timepoint","timepoint","TP","tp","time_point"), names(pheno_raw))[1]
trait_col <- colnames(pheno_raw)[19]

model_data <- pheno_raw %>%
  select(all_of(c(gid_col, env_col, tp_col, unname(band_cols), trait_col))) %>%
  rename(Env=all_of(env_col)) %>% rename(all_of(band_cols)) %>%
  filter(if_all(all_of(names(band_cols)), ~!is.na(.))) %>%
  mutate(Env=as.factor(Env), Timepoint=as.integer(.data[[tp_col]]),
         group=paste(Env, Timepoint, sep="_TP")) %>%
  rename(GID=all_of(gid_col)) %>%
  filter(group %in% c("HELF24_TP7","KET21_TP4","MCG23_TP5","MCG25_TP12","SNY22_TP6")) %>%
  mutate(NDVI_raw = (nir-red)/(nir+red+1e-8)) %>%
  filter(Env != "MCG25") %>%
  group_by(Env) %>%
  mutate(across(all_of(names(band_cols)), ~(. - mean(., na.rm=TRUE))/(sd(., na.rm=TRUE)+1e-8))) %>%
  ungroup()

BAND_NAMES <- names(band_cols)
env_levels <- sort(unique(as.character(model_data$Env)))
trait <- trait_col

# ── Helpers ───────────────────────────────────────────────────────────────
scale_from_train <- function(df, train_df, trait) {
  band_sc  <- map(BAND_NAMES, function(b)
    list(mn=min(train_df[[b]],na.rm=TRUE), mx=max(train_df[[b]],na.rm=TRUE))) %>%
    set_names(BAND_NAMES)
  trait_mn <- mean(train_df[[trait]], na.rm=TRUE)
  trait_sd <- sd(train_df[[trait]],   na.rm=TRUE)
  for (b in BAND_NAMES) {
    sc <- band_sc[[b]]
    df[[b]] <- (df[[b]]-sc$mn)/(sc$mx-sc$mn+1e-8)
  }
  df[[trait]] <- (df[[trait]]-trait_mn)/(trait_sd+1e-8)
  list(df=df, trait_mn=trait_mn, trait_sd=trait_sd)
}

make_poly_features <- function(df) {
  X <- as.matrix(df[, BAND_NAMES])
  sq <- X^2; colnames(sq) <- paste0(BAND_NAMES,"_sq")
  pairs <- combn(seq_len(ncol(X)),2,function(i) X[,i[1]]*X[,i[2]])
  colnames(pairs) <- combn(BAND_NAMES,2,paste,collapse="x")
  cbind(X, sq, pairs)
}

make_tensors <- function(df, trait) {
  df_sub <- df[!is.na(df[[trait]]),]
  list(bands=torch_tensor(as.matrix(df_sub[,BAND_NAMES,drop=FALSE]),dtype=torch_float()),
       y=torch_tensor(df_sub[[trait]],dtype=torch_float())$unsqueeze(2))
}

# ── NDVI LOEO baseline ────────────────────────────────────────────────────
cat("\n── NDVI linear LOEO ─────────────────────────────────────\n")
envs <- sort(unique(as.character(model_data$Env)))
ndvi_loeo <- map_dfr(envs, function(held) {
  tr <- model_data %>% filter(Env!=held, !is.na(.data[[trait]]))
  te <- model_data %>% filter(Env==held,  !is.na(.data[[trait]]))
  fit  <- lm(as.formula(paste(trait,"~ NDVI_raw")), data=tr)
  pred <- predict(fit, newdata=te)
  tibble(held_env=held, r=cor(pred,te[[trait]],use="complete.obs"))
})
cat(sprintf("NDVI LOEO mean r = %.3f\n", mean(ndvi_loeo$r)))
print(ndvi_loeo)

# ── LASSO polynomial SR LOEO ──────────────────────────────────────────────
cat("\n── LASSO polynomial SR LOEO ──────────────────────────────\n")
lasso_loeo <- map_dfr(envs, function(held) {
  train_raw <- model_data %>% filter(Env!=held, !is.na(.data[[trait]]))
  test_raw  <- model_data %>% filter(Env==held,  !is.na(.data[[trait]]))
  sc  <- scale_from_train(train_raw, train_raw, trait)
  tr  <- sc$df; trait_mn <- sc$trait_mn; trait_sd <- sc$trait_sd
  te  <- scale_from_train(test_raw, train_raw, trait)$df
  X_tr <- make_poly_features(tr); X_te <- make_poly_features(te)
  set.seed(42)
  fit    <- cv.glmnet(X_tr, tr[[trait]], alpha=1, nfolds=5)
  pred_z <- as.numeric(predict(fit, X_te, s="lambda.min"))
  pred   <- pred_z*trait_sd+trait_mn
  obs    <- test_raw[[trait]]
  coef_v <- as.numeric(coef(fit,s="lambda.min"))
  nz_terms <- sum(coef_v[-1]!=0)
  tibble(held_env=held, r=cor(pred,obs,use="complete.obs"), n_terms=nz_terms)
})
cat(sprintf("LASSO poly SR LOEO mean r = %.3f\n", mean(lasso_loeo$r)))
print(lasso_loeo)

# ── FullDNN LOEO (1 rep, 150 epochs — quick preview) ──────────────────────
cat("\n── FullDNN (no bottleneck) LOEO — quick preview (150 ep, 1 rep) ──\n")

FullDNNModel <- nn_module(
  classname = "FullDNNModel",
  initialize = function(n_bands=5, n_envs=5, d_embed=4L, h1=16L, h2=8L, pred_h=8L) {
    self$fc1      <- nn_linear(n_bands, h1)
    self$fc2      <- nn_linear(h1, h2)
    self$env_emb  <- nn_embedding(n_envs, d_embed)
    self$pred_fc1 <- nn_linear(h2+d_embed, pred_h)
    self$pred_fc2 <- nn_linear(pred_h, 1L)
    self$drop     <- nn_dropout(0.1)
  },
  forward = function(bands, env_idx=NULL, env_vec=NULL) {
    x <- bands %>% self$fc1() %>% nnf_relu() %>% self$drop()
    x <- x %>% self$fc2() %>% nnf_relu() %>% self$drop()
    ev <- if (!is.null(env_idx)) self$env_emb(env_idx) else
          env_vec$expand(c(x$size(1),-1L))
    torch_cat(list(x,ev),dim=2) %>% self$pred_fc1() %>% nnf_relu() %>%
      self$drop() %>% self$pred_fc2()
  }
)

train_quick <- function(train_df, trait, env_levels, epochs=150, lr=3e-3,
                         batch_size=64, seed=100L) {
  torch_manual_seed(seed)
  df_sub  <- train_df[!is.na(train_df[[trait]]),]
  tensors <- make_tensors(train_df, trait)
  n       <- tensors$bands$size(1)
  env_idx <- torch_tensor(match(as.character(df_sub$Env),env_levels),dtype=torch_long())
  model   <- FullDNNModel(length(BAND_NAMES),length(env_levels))
  opt     <- optim_adam(model$parameters,lr=lr,weight_decay=1e-4)
  sched   <- lr_step(opt,step_size=100,gamma=0.5)
  model$train()
  for (ep in seq_len(epochs)) {
    idx <- sample(n)
    for (i in seq(1,n,by=batch_size)) {
      bi <- idx[i:min(i+batch_size-1,n)]
      opt$zero_grad()
      pred <- model(tensors$bands[bi,,drop=FALSE],env_idx=env_idx[bi])
      loss <- nnf_mse_loss(pred,tensors$y[bi,,drop=FALSE])
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters,1.0)
      opt$step()
    }
    sched$step()
  }
  model$eval(); model
}

fulldnn_loeo <- map_dfr(envs, function(held) {
  train_raw <- model_data %>% filter(Env!=held, !is.na(.data[[trait]]))
  test_raw  <- model_data %>% filter(Env==held,  !is.na(.data[[trait]]))
  sc  <- scale_from_train(train_raw,train_raw,trait)
  tr  <- sc$df; trait_mn <- sc$trait_mn; trait_sd <- sc$trait_sd
  te  <- scale_from_train(test_raw,train_raw,trait)$df
  model <- train_quick(tr,trait,env_levels)
  train_env_idx <- torch_tensor(
    match(sort(unique(as.character(train_raw$Env))),env_levels),dtype=torch_long())
  mean_ev <- with_no_grad(model$env_emb(train_env_idx)$mean(dim=1,keepdim=TRUE))
  te_ten  <- make_tensors(te,trait)
  with_no_grad({
    pz <- model(te_ten$bands,env_vec=mean_ev)$squeeze()$to(device="cpu")
  })
  pred <- as.numeric(pz)*trait_sd+trait_mn
  obs  <- as.numeric(te_ten$y$squeeze())*trait_sd+trait_mn
  tibble(held_env=held, r=cor(pred,obs,use="complete.obs"))
})
cat(sprintf("FullDNN LOEO mean r = %.3f (preview; full render uses 300 ep × 3 rep)\n",
            mean(fulldnn_loeo$r)))
print(fulldnn_loeo)

# ── Summary table ─────────────────────────────────────────────────────────
cat("\n═══ LOEO Pearson r — model comparison ═══\n")
comparison <- ndvi_loeo %>% rename(NDVI=r) %>%
  left_join(lasso_loeo %>% select(held_env,LASSO_poly_SR=r), by="held_env") %>%
  left_join(fulldnn_loeo %>% select(held_env,FullDNN_preview=r), by="held_env")
print(comparison, digits=3)
cat(sprintf("\nMean r:\n  NDVI linear   = %.3f\n  LASSO poly SR = %.3f\n  FullDNN (150ep)= %.3f\n",
            mean(comparison$NDVI), mean(comparison$LASSO_poly_SR),
            mean(comparison$FullDNN_preview)))
