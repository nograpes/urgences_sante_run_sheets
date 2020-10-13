# Image files metadata.
suppressPackageStartupMessages(library(dplyr))
library(magrittr)
library(readr)
suppressPackageStartupMessages(library(sqldf))

can.be.numeric <- function(x) {
  !is.na(suppressWarnings(as.numeric(x)))
}

luhn_digit <- function(x) {
  padded <- sprintf("%08d", as.integer(x))
  spl <- do.call(rbind, strsplit(padded, ""))
  class(spl) <- "integer"
  check.digit <- spl[,ncol(spl),drop = FALSE]
  spl <- spl[,-ncol(spl), drop = FALSE]
  spl <- t(spl[,ncol(spl):1, drop = FALSE])
  dbl <- spl * rep(c(2,1), length.out = nrow(spl))
  squish <- ifelse(dbl >= 10, dbl - 9, dbl)
  colSums(squish) %% 10
}

actual.digit <- function(x) {
  padded <- sprintf("%08d", as.integer(x))
  spl <- do.call(rbind, strsplit(padded, ""))
  class(spl) <- "integer"
  spl[,ncol(spl)]
}

valid.luhn <- function(x) {
  padded <- sprintf("%08d", as.integer(x))
  spl <- do.call(rbind, strsplit(padded, ""))
  class(spl) <- "integer"
  check.digit <- spl[,ncol(spl),drop=FALSE]
  spl <- spl[,-ncol(spl),drop=FALSE]
  spl <- t(spl[,ncol(spl):1,drop=FALSE])
  dbl <- spl * rep(c(2,1), length.out = nrow(spl))
  squish <- ifelse(dbl >= 10, dbl - 9, dbl)
  true.digit <- colSums(squish) %% 10
  check.digit == true.digit
}

data.dir <- "D:/urgence_sante/run_sheets_full/R48_RIP"
if(Sys.info()['sysname'] == "Linux") {
  data.dir <- "/data/run_sheets_full/R48_RIP"
}

all.files <- list.files(data.dir, pattern = "*.png", full.names = TRUE, recursive = TRUE)

file.df <- 
  data.frame(
    file.path = all.files,
    stringsAsFactors = FALSE
  )

idx <- c(1:2 + length(strsplit(data.dir, "/")[[1]]))
file.df[c("NoCd", "rip")] <- 
  t(sapply(strsplit(file.df$file.path, "/"), `[`, idx))
file.df$file <- basename(file.df$file.path)
file.df$NoRIP <- gsub(".png", "", file.df$file)

# Map the RIP number now.
rip_correction <- 
  read_csv("rip_correction.csv",
           col_types = cols(
             NoRIP = col_character(),
             true_rip = col_character()
           ))

file.df <- 
  file.df %>% 
  left_join(rip_correction %>% mutate(NoRIP2 = NoRIP), by = "NoRIP") %>% 
  mutate(true_rip = ifelse(is.na(NoRIP2), NoRIP, true_rip)) %>% 
  select(-NoRIP2)

file.df$valid_luhn <- rep(NA, nrow(file.df))
nums <- can.be.numeric(file.df$true_rip)
file.df$valid_luhn[nums] <- valid.luhn(file.df$true_rip[nums])

# Map the sheet type.
unknown_rips_to_sheet_type <- 
  read_csv("unknown_rips_to_sheet_type.csv",
           col_types = cols(
             NoRIP = col_character(),
             print_run = col_character(),
             guess = col_character()
           ))

rip_to_print_run <- 
  read_csv("rip_to_print_run.csv",
           col_types = cols(
             print_run = col_character(),
             start_rip = col_integer(),
             end_rip = col_integer()
           ))

print_run_to_sheet_type <- 
  read_csv("print_run_to_sheet_type.csv",
           col_types = cols(
             sheet_type = col_character(),
             print_run = col_character()
           ))

detailed_rip_to_print_run <-
  read_csv("detailed_rip_to_print_run.csv",
           col_types = cols(
             NoCd = col_character(),
             rip = col_character(),
             NoRIP = col_character(),
             print_run = col_character()
           ))

# Special mapping where true_rip unknown.
unknown_rip <- 
  file.df %>% 
  filter(is.na(true_rip)) %>% 
  inner_join(unknown_rips_to_sheet_type, by = "NoRIP") %>% 
  inner_join(print_run_to_sheet_type, by = "print_run") %>% 
  select(-guess)

# Special exception where there are two sheets with the same 
# number but a different print_run.
exception_rip <- 
  file.df %>% 
  inner_join(detailed_rip_to_print_run,
             by = c("NoCd", "rip", "NoRIP")) %>% 
  inner_join(print_run_to_sheet_type, by = "print_run") 

# The vast majority fall into ranges.
known_rip <- 
  file.df %>% 
  filter(!is.na(true_rip)) %>% 
  anti_join(detailed_rip_to_print_run,
            by = c("NoCd", "rip", "NoRIP"))

known_rip <- 
  sqldf("select *
         from known_rip x
         join rip_to_print_run y
         on x.true_rip >= y.start_rip  
         and x.true_rip <= y.end_rip;") %>% 
  as_tibble %>% 
  select(-start_rip, -end_rip) %>% 
  left_join(print_run_to_sheet_type, by = "print_run")

file.df <- bind_rows(unknown_rip, exception_rip, known_rip)
write_csv(file.df, "checkboxes/file_df.csv")