library(tidyr)
suppressPackageStartupMessages(library(dplyr))
library(magrittr)
library(reshape2)
library(ggplot2)

# 3 sources of data, SIRAQ, RAO, images

# RIP2 RIP3 form version switched over.

# SIRAQ - Database for Quality Assurance
# Human-entered info in addition to box-detected

# CAD/RAO - Computer-aided dispatch/ x-assiste-ordinateur Software used for dispatching, time-of-arrival (dhArriveeLieux), the paramedic numbers (MatriculeParamedic1, 2)
# In the RAO, the primary key is the event number and the vehicle number.
# Event number is sometimes called call number.
# Multiple run sheets can be linked to one event number/vehicle number.
#   - If there are multiple patients from one call, for example.
# Time-of-arrival and likely one other time-related thing is from the RAO.
# That means that NoEven is probably not unique, it is probably only unique in
# combination with the NoVehicule.
# The best identifier then is NoCD + NoRIP

# Scanned images
# Seem to just be identified by the NoCD (the "edir" and the NoRIP (the filename)

# Approximately 415,000 have only an image (no data in the SIRAQ) They are coming.
# 279,690/354,922 had data in the SIRAQ.
# 22,130/0 had data in the SIRAQ, but no event number/vehicle number -- so no RAO data.
# 0/3 had data in the SIRAQ, but no image.
# Of the remaining, there is some problem with the image copy.

# When linking to the RAO, there were some NoEven + NoVehicule that were not
# found in the RAO (14,347 / 35,240). So we have these, but they should be
# missing (dhArriveeLieux, MatriculeParamedic1, 2). These are marked with
# JonctionRAO being 'O'

# New run sheets data.
# /data/run_sheets_full
# Highly compressible, but that might not be the point.
data.dir <- "D:/urgence_sante/run_sheets_full/R48_RIP"

# SIRAQ+RAO CSVs
rip2 <- read.csv(file.path(data.dir, "rip2.csv"), sep = ";", stringsAsFactors = FALSE,
                 colClasses = "character")
rip3 <- read.csv(file.path(data.dir, "rip3.csv"), sep = ";", 
                 colClasses = "character",
                 skipNul = TRUE,
                 stringsAsFactors = FALSE)

rip2$date <- as.Date(rip2$DtAppel, format = "%d%b%Y")
rip3$date <- as.Date(rip3$DtAppel, format = "%d%b%Y")

rip2$arrival.date <- as.Date(rip2$dhArriveeLieux, format = "%d%b%Y")
rip3$arrival.date <- as.Date(rip3$dhArriveeLieux, format = "%d%b%Y")

# Image files metadata.
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

library(readr)
write_csv(file.df, "file_df.csv")
