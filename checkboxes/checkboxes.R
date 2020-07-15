dir <- "/data/run_sheets_full"
# dir <- "D:\\urgence_sante\\run_sheets_full"
file <- paste0(dir, "/R48_RIP/rip2.csv")

library(readr)
rip2 <- read_delim(file, delim = ";", progress = FALSE)

dir <- "D:\\urgence_sante\\run_sheets_full\\R48_RIP"
images <- list.files(dir, recursive = TRUE, pattern = "*.png")
grep(rip2$NoRIP[1], images, value = TRUE)

# This has Monitorage checked. Let's find what is checked.
colnames(rip2)[as.character(rip2[1,]) == "O"]

# Prefix "am" indicates "Antecedents medicuax"
antecedents_medicaux <- grep("^am", colnames(rip2), value = T)
# amIndAucun (wrong?) - Antecedents medicaux/Aucun
# amIndIconnu (wrong?) - Antecedents medicaux/Inconnu
 
# Prefix "a" is the allergies box.
allergies <- grep("^aInd", colnames(rip2), value = T)

grep("^m", colnames(rip2), value = T)

# Prefix "i" must be the interventions
interventions <- grep("^i", colnames(rip2), value = T)
interventions[as.character(rip2[1,interventions]) == "O"]

# Prefix "t" is the transport. Somehow very good.
transport <- grep("^t", colnames(rip2), value = T)
transport[as.character(rip2[1,transport]) == "O"]

# Prefix "s" is the signes vitaux et traitements.
# These are all the checkboxes except AVPU and GCS
signes_vitaux <- grep("^sInd", colnames(rip2), value = T)
# There are five sets of each

# AVPU are radio boxes (does not include Oriente). Great!
signes_vitaux_avpu <- grep("^sAVPU", colnames(rip2), value = T)

# Oriente (Checkbox)
oriente <- grep("^sIndOriente", colnames(rip2), value = T)

# GCS (number)
gcs <- grep("^sGCS", colnames(rip2), value = T)

# Use a neural network to detect O2.
# O2 is what iInd02



grep("1", signes_vitaux, value = T)
signes_vitaux_AVPU <- grep("^s", colnames(rip2), value = T)
rip2[,oriente[1]]

# AVPU
# GCS
# Prefix "s" must indicate signes vitaux et traitements.
# sIndRSup4 