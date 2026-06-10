# ─────────────────────────────────────────────────────────────────────────────
# Save model CV results from knitr cache to data/model_results.Rdata
#
# Run from the project root:
#   source("scripts/save_model_results.R")
#
# Reads all heavy CV results directly from the knitr chunk cache (no
# re-training). After running, load in any script or Rmd with:
#   load("data/model_results.Rdata")
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages(library(tidyverse))

cache <- "analysis/DNN_Synthetic_VI_cache/html"
if (!dir.exists(cache)) stop("Cache directory not found. Render the Rmd first.")

# ── Helper: load a cache chunk by name prefix ─────────────────────────────
load_chunk <- function(prefix) {
  f <- list.files(cache, pattern = paste0("^", prefix, "_.*\\.rdb$"),
                  full.names = TRUE)
  if (!length(f)) { message("  [skip] no cache for: ", prefix); return(NULL) }
  e <- new.env()
  lazyLoad(sub("\\.rdb$", "", f[1]), envir = e)
  message("  [ok]   ", prefix)
  as.list(e)
}

message("Loading from knitr cache: ", cache)
tr   <- load_chunk("run-training")
lb   <- load_chunk("run-lbfgs")
ls   <- load_chunk("run-lasso-sr")
psr  <- load_chunk("run-pysr")
ts   <- load_chunk("run-two-stage")
ndvi <- load_chunk("ndvi-baseline")
cc   <- load_chunk("center-comparison")

# ── Assemble ──────────────────────────────────────────────────────────────
`%||%` <- function(a, b) if (!is.null(a)) a else b

cv_loeo            <- tr$cv_loeo
cv_wenv            <- tr$cv_wenv
env_levels         <- tr$env_levels
usable_traits      <- tr$usable_traits

cv_loeo_lbfgs      <- lb$cv_loeo_lbfgs
cv_wenv_lbfgs      <- lb$cv_wenv_lbfgs

cv_loeo_lasso      <- ls$cv_loeo_lasso
cv_wenv_lasso      <- ls$cv_wenv_lasso

cv_loeo_pysr       <- psr$cv_loeo_pysr  %||% tibble()
cv_wenv_pysr       <- psr$cv_wenv_pysr  %||% tibble()

ts_lbfgs           <- ts$ts_lbfgs
ts_lasso           <- ts$ts_lasso

ndvi_loeo          <- ndvi$ndvi_loeo    %||% tibble()
ndvi_wenv          <- ndvi$ndvi_wenv    %||% tibble()

cv_loeo_centered   <- cc$cv_loeo_centered    %||% tibble()
ndvi_loeo_centered <- cc$ndvi_loeo_centered  %||% tibble()

# ── Report ────────────────────────────────────────────────────────────────
objs <- list(
  cv_loeo            = cv_loeo,
  cv_wenv            = cv_wenv,
  cv_loeo_lbfgs      = cv_loeo_lbfgs,
  cv_wenv_lbfgs      = cv_wenv_lbfgs,
  cv_loeo_lasso      = cv_loeo_lasso,
  cv_wenv_lasso      = cv_wenv_lasso,
  cv_loeo_pysr       = cv_loeo_pysr,
  cv_wenv_pysr       = cv_wenv_pysr,
  ts_lbfgs           = ts_lbfgs,
  ts_lasso           = ts_lasso,
  ndvi_loeo          = ndvi_loeo,
  ndvi_wenv          = ndvi_wenv,
  cv_loeo_centered   = cv_loeo_centered,
  ndvi_loeo_centered = ndvi_loeo_centered
)
cat("\nObjects assembled:\n")
for (nm in names(objs)) {
  n <- if (is.data.frame(objs[[nm]])) nrow(objs[[nm]]) else length(objs[[nm]])
  cat(sprintf("  %-26s %d rows\n", nm, n))
}
cat(sprintf("  %-26s %s\n", "usable_traits", paste(usable_traits, collapse=", ")))
cat(sprintf("  %-26s %s\n", "env_levels",    paste(env_levels,    collapse=", ")))

# ── Save ─────────────────────────────────────────────────────────────────
out <- "data/model_results.Rdata"
save(cv_loeo, cv_wenv,
     cv_loeo_lbfgs,    cv_wenv_lbfgs,
     cv_loeo_lasso,    cv_wenv_lasso,
     cv_loeo_pysr,     cv_wenv_pysr,
     ts_lbfgs,         ts_lasso,
     ndvi_loeo,        ndvi_wenv,
     cv_loeo_centered, ndvi_loeo_centered,
     env_levels,       usable_traits,
     file = out)

cat(sprintf("\nSaved to %s (%.1f kB)\n", out, file.size(out) / 1024))
cat("Load in any Rmd or script with:\n  load(\"data/model_results.Rdata\")\n")
