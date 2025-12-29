#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(stringr)
})

main <- function(results_dir, output_path = "boxplot.png") {
  files <- list.files(results_dir, pattern = "test_metrics", full.names = TRUE)
  r2_cols <- c("r2")
  dfs <- list()
  print(files)

  for (f in files) {
    data <- read_csv(f, show_col_types = FALSE)

    # Extract the R2 values
    r2_values <- unlist(data[r2_cols], use.names = FALSE)
    print(r2_values)
    df_file <- data.frame(r2 = r2_values, stringsAsFactors = FALSE)

    file_name <- basename(f)
    parts <- str_split(file_name, "\\.", simplify = TRUE)

    df_file$feature_extractor <- parts[1]
    df_file$algorithm <- parts[2]

    dfs[[length(dfs) + 1]] <- df_file
  }

  df <- bind_rows(dfs)
  p <- ggplot(df, aes(x = feature_extractor, y = r2, fill = algorithm)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = "Feature extractor", y = "R2", title = "Benchmark")

  ggsave(output_path, plot = p, width = 15, height = 6, dpi = 300)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript boxplot_r2.R <results_dir> [output_path]", call. = FALSE)
}

results_dir <- args[1]
output_path <- if (length(args) >= 2) args[2] else "boxplot.png"

main(results_dir, output_path)
