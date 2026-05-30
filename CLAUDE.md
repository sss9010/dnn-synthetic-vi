# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An R analysis project comparing methods for learning data-optimised spectral vegetation indices from UAV multispectral imagery to predict barley grain yield across five field environments. The primary deliverable is a rendered HTML report.

## Rendering the Report

Run from the **project root** directory in R:

```r
rmarkdown::render("analysis/DNN_Synthetic_VI.Rmd",
                  output_dir = "docs",
                  output_file = "DNN_Synthetic_VI.html")
```

The knitr cache (`analysis/DNN_Synthetic_VI_cache/`) skips heavy training chunks (`run-training`, `run-adaptation-cv`, `run-fulldnn-training`, `run-lasso-sr`, `run-lbfgs`) if code is unchanged. Delete the cache directory to force a full re-run (~hours on CPU).

## Helper Scripts

All run from the project root:

```r
source("scripts/ndvi_corr.R")          # NDVI–yield Pearson r per environment
source("scripts/variance_partition.R") # eta² between- vs within-env variance decomposition
source("analysis/quick_model_comparison.R")  # fast LOEO preview (150 epochs, 1 rep, no MCG25)
source("analysis/extract_results.R")   # load and print results from knitr cache files
```

## Dependencies

```r
install.packages(c("tidyverse", "ggplot2", "patchwork", "corrplot",
                   "knitr", "kableExtra", "torch", "glmnet", "reticulate"))
torch::install_torch()   # one-time LibTorch download (~0.5 GB)
# PySR requires Python: reticulate::py_install("pysr")
```

## Architecture

### Data Flow

`data/WMB_pheno.Rdata` → column detection (flexible regex) → filter to 5 Env×Timepoint groups → NDVI outlier removal (±1.96 SD per env) → per-env z-score standardisation of bands → per-fold min-max scaling → models

The data file is looked up from multiple candidate paths (local clone, `data/WMB_pheno.Rdata`). Column names are detected by regex, not hardcoded, so the pipeline tolerates minor naming differences.

### Models Compared

| Model | Key object names |
|-------|-----------------|
| DNN bottleneck (5→16→8→1 encoder + env embed + head) | `cv_loeo`, `cv_wenv`, `cv_adaptation` |
| Full DNN (no bottleneck, 5→16→8 + env embed + head) | `cv_loeo_fulldnn`, `cv_wenv_fulldnn` |
| L-BFGS optimised normalised ratio VI + OLS | `cv_loeo_lbfgs`, `cv_wenv_lbfgs` |
| LASSO polynomial SR (degree-2, 20 terms) | `cv_loeo_lasso`, `cv_wenv_lasso` |
| PySR symbolic regression | `cv_loeo_pysr`, `cv_wenv_pysr` |
| NDVI linear regression (baseline) | `cv_loeo_ndvi`, `cv_wenv_ndvi` |

### Cross-Validation Designs

- **LOEO** — leave-one-environment-out (5 folds × 3 seeds)
- **Within-env 5-fold by GID** — splits by genotype ID within each environment (5 folds × 3 seeds)
- **Adapt B** — freeze DNN encoder, fine-tune head + embedding on adaptation GIDs
- **Adapt C** — freeze DNN encoder, fit OLS (trait ~ SynVI) on adaptation GIDs

Splits are always by GID to prevent data leakage. Scalers are fit on training data only and applied to test folds.

### DNN Architecture Detail

```
Band Encoder      5 bands → FC(16) → ReLU → dropout(0.1)
                                  → FC(8)  → ReLU → dropout(0.1)
                                  → FC(1)           ← Synthetic VI (linear, no activation)
                                       ↓
Prediction Head   [SynVI ‖ env_embed(4)] → FC(8) → ReLU → dropout → FC(1) → trait
```

For LOEO inference on held-out environments, the mean embedding of the four training environments is used as a proxy. The Full DNN variant omits the bottleneck (`FC(1)`) and concatenates the `FC(8)` output directly with the env embedding.

## Key Conventions

- Trait column is always `colnames(pheno_raw)[19]` — positional, not by name.
- Band columns are detected by regex (`^red$` vs `red[._]?edge|rededge` etc.) to avoid matching `red` inside `red_edge`.
- `model_data_raw` holds env-z-scored data before per-fold scaling (used in CV loops).
- `model_data_raw_orig` holds pre-filter data (used in MCG25 diagnostic plots).
- MCG25_TP12 is a late-season flight with an inverted spectral–yield relationship; it is deliberately retained to stress-test cross-environment generalisation.
- Performance metric is Pearson *r* on the back-transformed trait scale; RMSE is secondary.
