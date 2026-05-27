# DNN-Derived Synthetic Vegetation Index for Phenotype Prediction

A self-contained R analysis that trains a **deep neural network (DNN)** on
raw multispectral drone imagery to derive data-optimised *Synthetic Vegetation
Indices* (SynVI) for agronomic trait prediction in the NY Winter Malting Barley
breeding programme.

---

## What it does

Standard vegetation indices (NDVI, NDRE, …) use fixed, hand-crafted band
combinations. This analysis **lets the data find the optimal band combination**
for a target trait. The result is a trait-specific scalar index — the Synthetic
VI — computed from five raw spectral bands (blue, green, red, red-edge, NIR)
that maximises cross-validated predictive accuracy.

---

## Repository structure

```
dnn_synthetic_vi/
├── analysis/
│   └── DNN_Synthetic_VI.Rmd        # Main analysis (render this)
│   └── DNN_Synthetic_VI_cache/     # knitr cache for heavy training chunks
├── data/
│   └── WMB_pheno.Rdata             # Input: merged spectral + phenotype table
├── docs/
│   ├── DNN_Synthetic_VI.html       # Pre-rendered HTML report
│   └── figure/DNN_Synthetic_VI.Rmd/  # All generated figures (PNG)
└── scripts/
    ├── ndvi_corr.R                 # Quick NDVI–yield correlation summary
    └── variance_partition.R        # Between- vs within-env variance decomposition
```

---

## Data

`data/WMB_pheno.Rdata` contains a single data frame with:

- **Spectral bands** per plot: `blue`, `green`, `red`, `red_edge`, `nir`
- **Standard VIs**: NDVI, NDRE, EVI, GNDVI, etc.
- **Trait**: yield (column 19) and associated agronomic measurements
- **Metadata**: `GID` (genotype), `Env` (trial environment), `Timepoint`

Five Env × Timepoint groups are used (one optimal flight per trial year):  
`HELF24_TP7`, `KET21_TP4`, `MCG23_TP5`, `MCG25_TP12`, `SNY22_TP6`


---

## DNN architecture

```
Band Encoder   →  5 bands → [16] → [8] → 1 (Synthetic VI, no activation)
                                              ↓
Prediction Head → [SynVI | env_embedding(4)] → [8] → 1 (trait)
```

The bottleneck single-neuron output mimics the structure of hand-crafted indices
while learning the optimal non-linear band combination for each trait.

---

## Cross-validation schemes

| Scheme | Description |
|--------|-------------|
| **LOEO** (Leave-One-Environment-Out) | Train on 4 envs, predict 1 held-out env. Measures cross-environment generalisation. |
| **Within-env 5-fold by GID** | 5-fold CV within each environment, splits by genotype ID. Measures within-trial predictive ability. |

Both schemes use 3 random seed repetitions. NDVI linear regression runs under
the same designs as a baseline.

---

## R packages required

```r
install.packages(c(
  "tidyverse", "ggplot2", "patchwork", "corrplot",
  "knitr", "kableExtra",
  "torch"   # deep learning backend
))
torch::install_torch()   # one-time LibTorch download
```

---

## Rendering the report

Open R (or RStudio) from the **project root** and run:

```r
rmarkdown::render("analysis/DNN_Synthetic_VI.Rmd",
                  output_dir = "docs",
                  output_file = "DNN_Synthetic_VI.html")
```

The knitr cache in `analysis/DNN_Synthetic_VI_cache/` will skip the two heavy
training chunks (`run-training` and `run-adaptation-cv`) if the code is unchanged.
Delete the cache to force a full re-run.

---

## Helper scripts

Run from the project root:

```r
source("scripts/ndvi_corr.R")        # NDVI–yield Pearson r per environment
source("scripts/variance_partition.R") # eta² variance decomposition
```

---

## Results overview

The pre-rendered report is at `docs/DNN_Synthetic_VI.html`. Key outputs:

- **LOEO accuracy** — Pearson *r* per held-out environment
- **Within-env accuracy** — Pearson *r* per environment (5-fold by GID)
- **NDVI comparison** — DNN SynVI vs NDVI linear regression
- **Transfer learning** — two-stage adaptation CV (freeze encoder → fine-tune head)
- **Band sensitivity** — gradient-based spectral band importance per trait
- **Synthetic VI formula** — extracted encoder weights for direct computation

---

## Citation / contact

Siim Sepp — sss322cornell.edu 
