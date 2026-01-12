#!/usr/bin/env Rscript

library(ggplot2)
library(readr)
library(dplyr)
library(stringr)

df <- read.csv("summary.csv")
p <- ggplot(df, aes(x = feature_extractor, y = r2, fill = model)) +
  geom_boxplot() +
  theme_minimal() +
  ylim(0, 1) +
  labs(x = "Feature extractor", y = "R2", title = "Benchmark")

ggsave("boxplot.png", plot = p, width = 15, height = 6, dpi = 300)