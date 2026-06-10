# ─────────────────────────────────────────────────────────────────────────────
# Two-Stage VI Pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1  Learn a VI formula from N-1 training environments
#   A  L-BFGS normalised-ratio  (w_num · x) / (exp(w_den) · x)
#   B  DNN bottleneck encoder   5 → 16 → 8 → 1
#   D  LASSO polynomial         sparse degree-2 combination
#
# Stage 2  Calibrate with OLS(yield ~ VI) in a target environment
#   LOEO       VI from 4 envs; calibrate + test on held-out env GIDs
#   Within-env VI from 4 envs; calibrate + test within same env GIDs
#
# For both scenarios the VI is always learned from the N-1 OTHER environments,
# keeping Stage 1 and Stage 2 cleanly separated.
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(tidyverse)
  library(glmnet)
  library(torch)
})
if (!torch::torch_is_installed()) torch::install_torch()
set.seed(42); torch_manual_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════

pheno_paths <- c(
  "C:/Users/Siim Sepp/NY_WMBCL/data/WMB_pheno.Rdata",
  "~/Documents/GitHub/NY_WMBCL/data/WMB_pheno.Rdata",
  "data/WMB_pheno.Rdata"
)
pheno_file <- pheno_paths[file.exists(pheno_paths)][1]
if (is.na(pheno_file)) stop("WMB_pheno.Rdata not found.")
load(pheno_file)

loaded_objs  <- ls()
df_cands     <- loaded_objs[sapply(loaded_objs, function(x) is.data.frame(get(x)))]
sizes        <- sapply(df_cands, function(x) nrow(get(x)))
pheno_raw    <- get(df_cands[which.max(sizes)])

band_patterns <- c(blue="^blue$", green="^green$", red="^red$",
                   red_edge="red[._]?edge|rededge", nir="^nir$")
band_cols <- sapply(names(band_patterns), function(b) {
  m <- grep(band_patterns[b], names(pheno_raw), ignore.case=TRUE, value=TRUE)
  if (!length(m)) NA_character_ else m[1]
}, USE.NAMES=TRUE)
band_cols <- band_cols[!is.na(band_cols)]

env_col   <- intersect(c("Env","env","environment","trial"), names(pheno_raw))[1]
gid_col   <- intersect(c("GID","gid","Genotype","genotype","EntryName","entry_name",
                         "Line","line","Name","name"), names(pheno_raw))[1]
tp_col    <- intersect(c("Timepoint","timepoint","TP","tp","time_point"), names(pheno_raw))[1]
trait_col <- colnames(pheno_raw)[19]

model_data <- pheno_raw %>%
  select(all_of(c(gid_col, env_col, tp_col, unname(band_cols), trait_col))) %>%
  rename(Env = all_of(env_col)) %>%
  rename(all_of(band_cols)) %>%
  filter(if_all(all_of(names(band_cols)), ~ !is.na(.))) %>%
  mutate(Env       = as.factor(Env),
         Timepoint = as.integer(.data[[tp_col]]),
         group     = paste(Env, Timepoint, sep = "_TP")) %>%
  rename(GID = all_of(gid_col)) %>%
  filter(group %in% c("HELF24_TP7","KET21_TP4","MCG23_TP5","SNY22_TP6")) %>%
  mutate(NDVI_raw = (nir - red) / (nir + red + 1e-8)) %>%
  group_by(Env) %>%
  filter(abs(NDVI_raw - median(NDVI_raw, na.rm=TRUE)) <=
           2.5 * mad(NDVI_raw, na.rm=TRUE)) %>%
  mutate(across(all_of(names(band_cols)),
                ~ (. - mean(., na.rm=TRUE)) / (sd(., na.rm=TRUE) + 1e-8))) %>%
  ungroup()

BAND_NAMES  <- names(band_cols)
TRAIT       <- trait_col
ENV_LEVELS  <- sort(unique(as.character(model_data$Env)))

cat("Environments:", paste(ENV_LEVELS, collapse=", "), "\n")
cat("Trait column:", TRAIT, "\n")
cat("Rows after filtering:", nrow(model_data), "\n\n")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Scale bands to [0,1] using statistics from ref_df
scale_bands_01 <- function(df, ref_df) {
  for (b in BAND_NAMES) {
    mn <- min(ref_df[[b]], na.rm=TRUE)
    mx <- max(ref_df[[b]], na.rm=TRUE)
    df[[b]] <- (df[[b]] - mn) / (mx - mn + 1e-8)
  }
  df
}

# Z-score trait using statistics from ref_df; return df + back-transform params
scale_trait_z <- function(df, ref_df, trait) {
  mn <- mean(ref_df[[trait]], na.rm=TRUE)
  sd <- sd(ref_df[[trait]],   na.rm=TRUE)
  df[[trait]] <- (df[[trait]] - mn) / (sd + 1e-8)
  list(df=df, mn=mn, sd=sd)
}

# Degree-2 polynomial features: 5 linear + 5 squared + 10 pairwise = 20 terms
poly_features <- function(df) {
  X  <- as.matrix(df[, BAND_NAMES])
  sq <- X^2; colnames(sq) <- paste0(BAND_NAMES, "_sq")
  pr <- combn(seq_len(ncol(X)), 2, function(i) X[,i[1]] * X[,i[2]])
  colnames(pr) <- combn(BAND_NAMES, 2, paste, collapse="x")
  cbind(X, sq, pr)
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. STAGE 1 — VI LEARNERS
# ══════════════════════════════════════════════════════════════════════════════
# Each learner receives a scaled training data frame (bands [0,1], trait z-scored)
# and returns vi_fn: function(X_matrix) -> numeric vector
# X_matrix columns must match BAND_NAMES and use the same [0,1] band scaling.

# ── A: L-BFGS normalised-ratio VI ─────────────────────────────────────────
# VI = (w_num · x) / (exp(w_den) · x)  initialised at the NDVI solution.
# Maximises |cor(VI, yield)| on training data.

learn_lbfgs_vi <- function(train_sc, trait) {
  n   <- length(BAND_NAMES)
  ni  <- which(BAND_NAMES == "nir")
  ri  <- which(BAND_NAMES == "red")
  init_num <- rep(0,  n); init_num[ni] <-  1; init_num[ri] <- -1
  init_den <- rep(-3, n); init_den[ni] <-  0; init_den[ri] <-  0

  X <- as.matrix(train_sc[!is.na(train_sc[[trait]]), BAND_NAMES])
  y <- train_sc[[trait]][!is.na(train_sc[[trait]])]

  obj <- function(p) {
    w_num <- p[seq_len(n)]
    w_den <- exp(p[seq_len(n) + n])
    vi    <- drop(X %*% w_num) / (drop(X %*% w_den) + 1e-8)
    if (any(!is.finite(vi)) || sd(vi) < 1e-6) return(1.0)
    -abs(cor(vi, y))
  }
  opt <- tryCatch(
    optim(c(init_num, init_den), obj, method="L-BFGS-B",
          control=list(maxit=500)),
    error = function(e) NULL
  )
  if (is.null(opt)) {
    warning("L-BFGS optimisation failed; returning NDVI.")
    w_num_fit <- init_num
    w_den_fit <- exp(init_den)
  } else {
    w_num_fit <- opt$par[seq_len(n)]
    w_den_fit <- exp(opt$par[seq_len(n) + n])
  }
  formula_str <- sprintf(
    "(%s) / (%s)",
    paste(sprintf("%+.4f*%s", w_num_fit, BAND_NAMES), collapse=" "),
    paste(sprintf("+%.4f*%s",  w_den_fit, BAND_NAMES), collapse=" ")
  )
  vi_fn <- function(X_mat) drop(X_mat %*% w_num_fit) / (drop(X_mat %*% w_den_fit) + 1e-8)
  attr(vi_fn, "formula") <- formula_str
  vi_fn
}

# ── B: DNN bottleneck VI ───────────────────────────────────────────────────
# Trains SyntheticVIModel (5→16→8→1 encoder + env embedding + head).
# Returns the frozen encoder as vi_fn; the head and embedding are discarded
# after Stage 1 — Stage 2 uses only the single bottleneck scalar.

BandEncoder <- nn_module(
  classname = "BandEncoder",
  initialize = function(n_bands=5, h1=16L, h2=8L) {
    self$fc1  <- nn_linear(n_bands, h1)
    self$fc2  <- nn_linear(h1, h2)
    self$fc3  <- nn_linear(h2, 1L)
    self$drop <- nn_dropout(0.1)
  },
  forward = function(x) {
    x <- x %>% self$fc1() %>% nnf_relu() %>% self$drop()
    x <- x %>% self$fc2() %>% nnf_relu() %>% self$drop()
    self$fc3(x)
  }
)

SyntheticVIModel <- nn_module(
  classname = "SyntheticVIModel",
  initialize = function(n_bands=5, n_envs=5, d_embed=4L, pred_h=8L) {
    self$encoder  <- BandEncoder(n_bands)
    self$env_emb  <- nn_embedding(n_envs, d_embed)
    self$pred_fc1 <- nn_linear(1L + d_embed, pred_h)
    self$pred_fc2 <- nn_linear(pred_h, 1L)
    self$drop     <- nn_dropout(0.1)
  },
  forward = function(bands, env_idx=NULL, env_vec=NULL) {
    vi <- self$encoder(bands)
    ev <- if (!is.null(env_idx)) self$env_emb(env_idx$view(-1L)) else
          env_vec$expand(c(vi$size(1), -1L))
    torch_cat(list(vi, ev), dim=2) %>%
      self$pred_fc1() %>% nnf_relu() %>% self$drop() %>%
      self$pred_fc2()
  }
)

learn_dnn_vi <- function(train_sc, trait, train_env_levels,
                          epochs=300, lr=3e-3, batch_size=64, seed=42L) {
  torch_manual_seed(seed)
  df_sub  <- train_sc[!is.na(train_sc[[trait]]), ]
  bands_t <- torch_tensor(as.matrix(df_sub[, BAND_NAMES]), dtype=torch_float())
  y_t     <- torch_tensor(df_sub[[trait]], dtype=torch_float())$unsqueeze(2)
  env_idx <- torch_tensor(match(as.character(df_sub$Env), train_env_levels),
                          dtype=torch_long())
  n       <- bands_t$size(1)

  model <- SyntheticVIModel(n_bands=length(BAND_NAMES),
                             n_envs=length(train_env_levels))
  opt   <- optim_adam(model$parameters, lr=lr, weight_decay=1e-4)
  sched <- lr_step(opt, step_size=100, gamma=0.5)

  model$train()
  for (ep in seq_len(epochs)) {
    idx <- sample(n)
    for (i in seq(1, n, by=batch_size)) {
      bi <- idx[i:min(i + batch_size - 1, n)]
      opt$zero_grad()
      pred <- model(bands_t[bi,,drop=FALSE], env_idx=env_idx[bi])
      loss <- nnf_mse_loss(pred, y_t[bi,,drop=FALSE])
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, 1.0)
      opt$step()
    }
    sched$step()
  }
  model$eval()

  # Freeze encoder; discard head and embedding
  encoder_state <- model$encoder$state_dict()
  frozen_encoder <- BandEncoder(n_bands=length(BAND_NAMES))
  frozen_encoder$load_state_dict(encoder_state)
  frozen_encoder$eval()

  function(X_mat) {
    bt <- torch_tensor(X_mat, dtype=torch_float())
    with_no_grad({
      vi <- frozen_encoder(bt)$squeeze()
      if (vi$dim() == 0L) vi <- vi$unsqueeze(1L)
      as.numeric(vi$to(device="cpu"))
    })
  }
}

# ── B-ZS: DNN full model (zero-shot, no Stage 2 calibration) ──────────────
# Same Stage 1 training as B. Returns the full frozen SyntheticVIModel so
# the prediction head + mean-embedding proxy can be used directly in Stage 2
# without any OLS calibration step.

learn_dnn_full <- function(train_sc, trait, train_env_levels,
                            epochs=300, lr=3e-3, batch_size=64, seed=42L) {
  torch_manual_seed(seed)
  df_sub  <- train_sc[!is.na(train_sc[[trait]]), ]
  bands_t <- torch_tensor(as.matrix(df_sub[, BAND_NAMES]), dtype=torch_float())
  y_t     <- torch_tensor(df_sub[[trait]], dtype=torch_float())$unsqueeze(2)
  env_idx <- torch_tensor(match(as.character(df_sub$Env), train_env_levels),
                          dtype=torch_long())
  n <- bands_t$size(1)

  model <- SyntheticVIModel(n_bands=length(BAND_NAMES),
                             n_envs=length(train_env_levels))
  opt   <- optim_adam(model$parameters, lr=lr, weight_decay=1e-4)
  sched <- lr_step(opt, step_size=100, gamma=0.5)

  model$train()
  for (ep in seq_len(epochs)) {
    idx <- sample(n)
    for (i in seq(1, n, by=batch_size)) {
      bi <- idx[i:min(i + batch_size - 1, n)]
      opt$zero_grad()
      pred <- model(bands_t[bi,,drop=FALSE], env_idx=env_idx[bi])
      loss <- nnf_mse_loss(pred, y_t[bi,,drop=FALSE])
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, 1.0)
      opt$step()
    }
    sched$step()
  }
  model$eval()
  model
}

# ── D: LASSO polynomial VI ─────────────────────────────────────────────────
# Fits LASSO on degree-2 polynomial features (20 terms).
# The LASSO prediction in training-trait-z units serves as the VI scalar.

learn_lasso_vi <- function(train_sc, trait) {
  X_tr <- poly_features(train_sc[!is.na(train_sc[[trait]]), ])
  y_tr <- train_sc[[trait]][!is.na(train_sc[[trait]])]
  set.seed(42)
  fit <- cv.glmnet(X_tr, y_tr, alpha=1, nfolds=5)

  nz <- sum(as.numeric(coef(fit, s="lambda.min"))[-1] != 0)
  cat(sprintf("  LASSO VI: %d non-zero terms (lambda.min = %.4f)\n",
              nz, fit$lambda.min))

  function(X_mat) {
    Xp <- {
      df_tmp <- as.data.frame(X_mat); names(df_tmp) <- BAND_NAMES
      poly_features(df_tmp)
    }
    as.numeric(predict(fit, Xp, s="lambda.min"))
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. STAGE 2 — OLS CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
# Given a fixed vi_fn (learned in Stage 1 on N-1 environments):
#   1. Apply vi_fn to target environment bands (scaled with training stats)
#   2. GID-fold CV: fit lm(yield ~ VI) on train GIDs, predict test GIDs
# Yield is back on original scale; r and RMSE are in original units.

ols_vi_cv <- function(target_df, vi_fn, trait, band_scale_ref,
                       n_folds=5, n_reps=3) {
  target_sc  <- scale_bands_01(target_df, band_scale_ref)
  vi_vals    <- vi_fn(as.matrix(target_sc[, BAND_NAMES]))
  target_sc$VI <- vi_vals
  gids <- unique(target_sc$GID)
  if (length(gids) < n_folds) return(NULL)

  map_dfr(seq_len(n_reps), function(rep_i) {
    set.seed(rep_i * 77L)
    fold_id <- rep(seq_len(n_folds), length.out=length(gids))
    gid_fold <- tibble(GID=sample(gids), fold=fold_id)
    map_dfr(seq_len(n_folds), function(f) {
      test_gids <- gid_fold$GID[gid_fold$fold == f]
      tr <- target_sc %>% filter(!GID %in% test_gids, !is.na(.data[[trait]]))
      te <- target_sc %>% filter( GID %in% test_gids,  !is.na(.data[[trait]]))
      if (nrow(tr) < 5 || nrow(te) < 2) return(NULL)
      fit  <- lm(as.formula(paste(trait, "~ VI")), data=tr)
      pred <- predict(fit, newdata=te)
      obs  <- te[[trait]]
      tibble(fold=f, rep=rep_i, n_test=nrow(te),
             r    = cor(pred, obs, use="complete.obs"),
             rmse = sqrt(mean((pred - obs)^2, na.rm=TRUE)))
    })
  })
}

# ── Zero-shot evaluation (no calibration) ────────────────────────────────
# Applies the full frozen model to the target environment using the mean
# embedding of training environments as a proxy. No GID splits needed —
# predictions are made on all target observations in one pass.

zero_shot_eval <- function(target_df, full_model, train_env_levels,
                            band_scale_ref, trait, trait_mn, trait_sd) {
  target_sc <- scale_bands_01(target_df, band_scale_ref)
  keep      <- !is.na(target_sc[[trait]])
  X         <- torch_tensor(as.matrix(target_sc[keep, BAND_NAMES]),
                            dtype=torch_float())
  y_obs     <- target_df[[trait]][keep]

  train_idx <- torch_tensor(seq_along(train_env_levels), dtype=torch_long())
  mean_ev   <- with_no_grad(
    full_model$env_emb(train_idx)$mean(dim=1, keepdim=TRUE)
  )
  with_no_grad({
    pred_z <- full_model(X, env_vec=mean_ev)$squeeze()$to(device="cpu")
  })
  pred <- as.numeric(pred_z) * trait_sd + trait_mn
  tibble(r    = cor(pred, y_obs, use="complete.obs"),
         rmse = sqrt(mean((pred - y_obs)^2, na.rm=TRUE)))
}

two_stage_cv_zeroshot <- function(data_raw, trait,
                                   stage1_epochs=300, stage1_lr=3e-3,
                                   stage1_seed=42L) {
  envs <- sort(unique(as.character(data_raw$Env)))
  map_dfr(envs, function(target_env) {
    cat(sprintf("  [DNN-ZS] target env: %s\n", target_env))
    train_raw  <- data_raw %>% filter(Env != target_env, !is.na(.data[[trait]]))
    target_raw <- data_raw %>% filter(Env == target_env, !is.na(.data[[trait]]))
    if (nrow(train_raw) < 10) return(NULL)

    train_sc_list    <- scale_trait_z(scale_bands_01(train_raw, train_raw),
                                      train_raw, trait)
    train_sc         <- train_sc_list$df
    trait_mn         <- train_sc_list$mn
    trait_sd         <- train_sc_list$sd
    train_env_levels <- sort(unique(as.character(train_sc$Env)))

    full_model <- tryCatch(
      learn_dnn_full(train_sc, trait, train_env_levels,
                     epochs=stage1_epochs, lr=stage1_lr, seed=stage1_seed),
      error = function(e) { message(e$message); NULL }
    )
    if (is.null(full_model)) return(NULL)

    zero_shot_eval(target_raw, full_model, train_env_levels,
                   band_scale_ref=train_raw, trait=trait,
                   trait_mn=trait_mn, trait_sd=trait_sd) %>%
      mutate(target_env=target_env, vi_formula=NA_character_, .before=1)
  })
}

# ══════════════════════════════════════════════════════════════════════════════
# 5. TWO-STAGE CV WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════
# For every environment E:
#   Stage 1  learn VI on the N-1 other environments
#   Stage 2  GID-fold CV with OLS(yield ~ VI) within E
#
# LOEO interpretation:   how well does the calibrated VI predict new genotypes
#                        in a completely unseen environment?
# Within-env interpretation: how well does the VI (learned elsewhere) predict
#                        new genotypes inside a known environment?

two_stage_cv <- function(data_raw, trait, vi_learner,
                          n_folds=5, n_reps=3,
                          stage1_epochs=300, stage1_lr=3e-3,
                          stage1_seed=42L, label="") {
  envs <- sort(unique(as.character(data_raw$Env)))
  map_dfr(envs, function(target_env) {
    cat(sprintf("  [%s] target env: %s\n", label, target_env))
    train_raw  <- data_raw %>% filter(Env != target_env, !is.na(.data[[trait]]))
    target_raw <- data_raw %>% filter(Env == target_env, !is.na(.data[[trait]]))
    if (nrow(train_raw) < 10 || nrow(target_raw) < n_folds) return(NULL)

    # Band [0,1] scaling + trait z-score from training environments only
    train_sc_list <- scale_trait_z(
      scale_bands_01(train_raw, train_raw), train_raw, trait
    )
    train_sc  <- train_sc_list$df
    train_env_levels <- sort(unique(as.character(train_sc$Env)))

    # Stage 1: learn VI
    vi_fn <- tryCatch(
      vi_learner(train_sc, trait,
                 train_env_levels = train_env_levels,
                 epochs = stage1_epochs,
                 lr     = stage1_lr,
                 seed   = stage1_seed),
      error = function(e) { message(e$message); NULL }
    )
    if (is.null(vi_fn)) return(NULL)
    vi_formula <- attr(vi_fn, "formula") %||% NA_character_

    # Stage 2: OLS calibration CV within target environment
    res <- ols_vi_cv(target_raw, vi_fn, trait,
                     band_scale_ref = train_raw,
                     n_folds = n_folds, n_reps = n_reps)
    if (is.null(res)) return(NULL)
    res %>% mutate(target_env = target_env, vi_formula = vi_formula, .before=1)
  })
}

# ══════════════════════════════════════════════════════════════════════════════
# 6. VI LEARNER WRAPPERS (absorb extra args cleanly)
# ══════════════════════════════════════════════════════════════════════════════

# NDVI: fixed formula, no learning — Stage 1 is a no-op
vi_ndvi <- function(train_sc, trait, train_env_levels=NULL,
                     epochs=NULL, lr=NULL, seed=NULL) {
  ni <- which(BAND_NAMES == "nir")
  ri <- which(BAND_NAMES == "red")
  vi_fn <- function(X_mat)
    (X_mat[, ni] - X_mat[, ri]) / (X_mat[, ni] + X_mat[, ri] + 1e-8)
  attr(vi_fn, "formula") <- "(nir - red) / (nir + red)"
  vi_fn
}

vi_lbfgs <- function(train_sc, trait, train_env_levels=NULL,
                      epochs=NULL, lr=NULL, seed=NULL)
  learn_lbfgs_vi(train_sc, trait)

vi_dnn <- function(train_sc, trait, train_env_levels,
                    epochs=300, lr=3e-3, seed=42L)
  learn_dnn_vi(train_sc, trait, train_env_levels,
               epochs=epochs, lr=lr, seed=seed)

vi_lasso <- function(train_sc, trait, train_env_levels=NULL,
                      epochs=NULL, lr=NULL, seed=NULL)
  learn_lasso_vi(train_sc, trait)

# ══════════════════════════════════════════════════════════════════════════════
# 7. RUN
# ══════════════════════════════════════════════════════════════════════════════

summarise_cv <- function(res, label) {
  if (is.null(res) || nrow(res) == 0) {
    cat(sprintf("\n%s: no results\n", label)); return(invisible(NULL))
  }
  summary <- res %>%
    group_by(target_env) %>%
    summarise(mean_r    = round(mean(r,    na.rm=TRUE), 3),
              mean_rmse = round(mean(rmse, na.rm=TRUE), 3),
              .groups   = "drop")
  cat(sprintf("\n══ %s ══\n", label))
  print(summary)
  cat(sprintf("  Overall mean r = %.3f\n", mean(summary$mean_r)))
  invisible(summary)
}

cat("\n═══ NDVI baseline (fixed formula, no Stage 1 learning) ═══\n")
res_ndvi <- two_stage_cv(model_data, TRAIT, vi_ndvi, label="NDVI")
summarise_cv(res_ndvi, "NDVI two-stage")

cat("\n═══ A: L-BFGS normalised-ratio VI ═══\n")
res_lbfgs <- two_stage_cv(model_data, TRAIT, vi_lbfgs, label="L-BFGS")
summarise_cv(res_lbfgs, "L-BFGS two-stage")

cat("\n── L-BFGS VI formula per held-out environment ──\n")
res_lbfgs %>%
  distinct(target_env, vi_formula) %>%
  arrange(target_env) %>%
  { cat(paste(sprintf("  %s:\n    %s\n", .$target_env, .$vi_formula),
              collapse="")); . } %>%
  invisible()

cat("\n═══ D: LASSO polynomial VI ═══\n")
res_lasso <- two_stage_cv(model_data, TRAIT, vi_lasso, label="LASSO")
summarise_cv(res_lasso, "LASSO two-stage")

cat("\n═══ B: DNN bottleneck VI + OLS (300 ep) ═══\n")
res_dnn <- two_stage_cv(model_data, TRAIT, vi_dnn,
                         stage1_epochs=300, label="DNN")
summarise_cv(res_dnn, "DNN two-stage (encoder + OLS)")

cat("\n═══ B-ZS: DNN full model, zero-shot (300 ep) ═══\n")
res_dnn_zs <- two_stage_cv_zeroshot(model_data, TRAIT, stage1_epochs=300)
summarise_cv(res_dnn_zs, "DNN zero-shot (full model, mean embedding)")

# ── Final comparison ──────────────────────────────────────────────────────
cat("\n═══ Summary: mean Pearson r by target environment ═══\n")
bind_rows(
  res_ndvi   %>% mutate(method="NDVI"),
  res_lbfgs  %>% mutate(method="A_LBFGS"),
  res_lasso  %>% mutate(method="D_LASSO"),
  res_dnn    %>% mutate(method="B_DNN_OLS"),
  res_dnn_zs %>% mutate(method="B_DNN_ZS")
) %>%
  group_by(method, target_env) %>%
  summarise(r = round(mean(r, na.rm=TRUE), 3), .groups="drop") %>%
  pivot_wider(names_from=method, values_from=r) %>%
  select(target_env, NDVI, A_LBFGS, D_LASSO, B_DNN_OLS, B_DNN_ZS) %>%
  print()
