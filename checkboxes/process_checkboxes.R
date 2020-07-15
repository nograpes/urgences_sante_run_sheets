# First do file_df.R
library(readr)
file.df <- 
  read_csv("checkboxes/file_df.csv", 
            col_types = cols(
              file.path = col_character(),
              NoCd = col_character(),
              rip = col_character(),
              file = col_character(),
              NoRIP = col_character()
            ),
           progress = FALSE)

my_copy <-
  read_csv("checkboxes/my_copy.csv",
           col_types = cols(
             .default = col_character(),
             sHrSV1 = col_time(format = ""),
             sHrSV2 = col_time(format = ""),
             sHrSV3 = col_time(format = ""),
             sHrSV4 = col_time(format = ""),
             sHrSV5 = col_time(format = ""),
             sResp1 = col_double(),
             sResp2 = col_double(),
             sResp3 = col_double(),
             sResp4 = col_double(),
             sResp5 = col_double(),
             sPouls1 = col_double(),
             sPouls2 = col_double(),
             sPouls3 = col_double(),
             sPouls4 = col_double(),
             sPouls5 = col_double(),
             sTASys1 = col_double(),
             sTASys2 = col_double(),
             sTASys3 = col_double(),
             sTASys4 = col_double(),
             sTASys5 = col_double(),
             sTADia1 = col_double(),
             sTADia2 = col_double(),
             sTADia3 = col_double(),
             sTADia4 = col_double(),
             sTADia5 = col_double(),
             sSpO21 = col_double(),
             sSpO22 = col_double(),
             sSpO23 = col_double(),
             sSpO24 = col_double(),
             sSpO25 = col_double()
           )
  )


library(dplyr)
df <- 
  my_copy %>% 
  left_join(file.df, by = "NoRIP") 

# Now do downsampling_checkboxes.