suppressMessages({ library(tidyverse) })

# Resolve data path relative to project root (run from project root, or adjust)
data_file <- here::here("data", "WMB_pheno.Rdata")
if (!file.exists(data_file)) data_file <- "data/WMB_pheno.Rdata"
load(data_file)

objs <- ls()
df   <- get(objs[sapply(objs, function(x) is.data.frame(get(x)))][1])

# ── Column detection ──────────────────────────────────────────────────────
band_patterns <- c(blue     = "^blue$",
                   green    = "^green$",
                   red      = "^red$",
                   red_edge = "red[._]?edge|rededge",
                   nir      = "^nir$")
band_cols <- sapply(names(band_patterns), function(b)
  grep(band_patterns[b], names(df), ignore.case = TRUE, value = TRUE)[1])
band_cols <- band_cols[!is.na(band_cols)]
env_col   <- intersect(c("Env","env","environment","trial"), names(df))[1]
tp_col    <- intersect(c("Timepoint","timepoint","TP","tp"),  names(df))[1]
trait     <- colnames(df)[19]

# ── Filter to optimal timepoints ─────────────────────────────────────────
df2 <- df %>%
  mutate(Timepoint = as.integer(.data[[tp_col]]),
         group = paste(.data[[env_col]], Timepoint, sep = "_TP")) %>%
  filter(group %in% c("HELF24_TP7","KET21_TP4","MCG23_TP5","MCG25_TP12","SNY22_TP6")) %>%
  rename(Env = all_of(env_col)) %>%
  rename(all_of(band_cols)) %>%
  mutate(NDVI = (nir - red) / (nir + red + 1e-8))

BAND_NAMES <- names(band_cols)
predictors <- c(BAND_NAMES, "NDVI", trait)

# ── One-way ANOVA variance decomposition ─────────────────────────────────
decomp <- map_dfr(predictors, function(v) {
  x   <- df2[[v]]
  env <- df2$Env
  ok  <- !is.na(x) & !is.na(env)
  x   <- x[ok]; env <- env[ok]
  N   <- length(x)

  grand_mean <- mean(x)
  env_means  <- tapply(x, env, mean)
  env_n      <- tapply(x, env, length)

  SS_total   <- sum((x - grand_mean)^2)
  SS_between <- sum(env_n * (env_means - grand_mean)^2)
  SS_within  <- SS_total - SS_between
  eta2       <- SS_between / SS_total

  sd_between <- sqrt(SS_between / (N - 1))
  sd_within  <- sqrt(SS_within  / (N - 1))

  tibble(
    Predictor   = v,
    N           = N,
    SD_total    = round(sqrt(SS_total / (N - 1)), 4),
    SD_between  = round(sd_between, 4),
    SD_within   = round(sd_within,  4),
    pct_between = round(100 * eta2,       1),
    pct_within  = round(100 * (1 - eta2), 1),
    eta2        = round(eta2, 4)
  )
})

cat("\n=== Variance Partitioning: Between vs. Within Environment ===\n\n")
print(as.data.frame(decomp), row.names = FALSE)
cat("\n\nNote: pct_between = eta² × 100 (proportion of total variance\n")
cat("explained by environment grouping in a one-way ANOVA).\n")
