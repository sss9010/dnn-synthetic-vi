suppressMessages({ library(tidyverse) })

# Resolve data path relative to project root (run from project root, or adjust)
data_file <- here::here("data", "WMB_pheno.Rdata")
if (!file.exists(data_file)) data_file <- "data/WMB_pheno.Rdata"
load(data_file)

objs    <- ls()
df_names <- objs[sapply(objs, function(x) is.data.frame(get(x)))]
df      <- get(df_names[1])

red_col <- grep("^red$",  names(df), ignore.case = TRUE, value = TRUE)[1]
nir_col <- grep("^nir$",  names(df), ignore.case = TRUE, value = TRUE)[1]
env_col <- intersect(c("Env","env","environment","trial"), names(df))[1]
tp_col  <- intersect(c("Timepoint","timepoint","TP","tp"), names(df))[1]
trait   <- colnames(df)[19]

cat("Cols: red =", red_col, "  nir =", nir_col,
    "  env =", env_col, "  trait =", trait, "\n")

df2 <- df %>%
  mutate(Timepoint = as.integer(.data[[tp_col]]),
         group = paste(.data[[env_col]], Timepoint, sep = "_TP")) %>%
  filter(group %in% c("HELF24_TP7","KET21_TP4","MCG23_TP5","MCG25_TP12","SNY22_TP6")) %>%
  mutate(NDVI = (.data[[nir_col]] - .data[[red_col]]) /
               (.data[[nir_col]] + .data[[red_col]] + 1e-8))

results <- df2 %>%
  filter(!is.na(NDVI), !is.na(.data[[trait]])) %>%
  group_by(Env = .data[[env_col]]) %>%
  summarise(
    n          = n(),
    r          = round(cor(NDVI, .data[[trait]], use = "complete.obs"), 3),
    mean_NDVI  = round(mean(NDVI,           na.rm = TRUE), 3),
    sd_NDVI    = round(sd(NDVI,             na.rm = TRUE), 3),
    mean_yield = round(mean(.data[[trait]], na.rm = TRUE), 1),
    sd_yield   = round(sd(.data[[trait]],   na.rm = TRUE), 1),
    .groups    = "drop"
  ) %>%
  arrange(desc(r))

cat("\nNDVI-Yield correlations per environment:\n")
print(as.data.frame(results))
