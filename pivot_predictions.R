# pivot predictions
library(readr)
uncoded_predictions_long <- 
  read_csv("checkboxes/uncoded_predictions_long.csv",
           col_types = cols(
             checkbox = col_logical(),
             path = col_character(),
             file = col_character(),
             big_file = col_character(),
             checkbox_name = col_character()
           ))

library(tidyr)
suppressPackageStartupMessages(library(dplyr))

uncoded_predictions <- 
  uncoded_predictions_long %>% 
  mutate(checkbox = ifelse(checkbox, "O", "N")) %>% m,.n ,
  mutate(NoRIP = tools::file_path_sans_ext(big_file)) %>% 
  pivot_wider(id_cols = NoRIP,
              names_from = checkbox_name,
              values_from = checkbox)

write_csv(uncoded_predictions, "checkboxes/uncoded_predictions.csv")
