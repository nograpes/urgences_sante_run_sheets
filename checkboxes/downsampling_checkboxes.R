# Do three types of downsampling.
library(imager)
library(parallel)
options(mc.cores = 20)

crop <- function(file, x, y) {
  im <- grayscale(load.image(file))
  imager:::as.cimg.array(im[x,y,,])
}

to_2d_array <- function(img) {
  as.array(img)[,,1,1]
}

downsample <- function(im, interp, scale) {
  if(length(dim(im)) == 2) dim(im) <- c(dim(im), 1, 1)
  to_2d_array(imresize(im, scale = scale, interp = interp))
}

crop.and.downsample <- function(file, x, y, interp = 3, scale = 0.3) {
  im <- grayscale(load.image(file))
  lr <- imager:::as.cimg.array(im[x,y,,])
  test <- imresize(lr, scale = scale, interp = interp)
  as.array(test)[,,1,1]
}

plot.array <- function(arr) {
  plot(imager:::as.cimg.array(arr))
}

plot.runsheet <- function(n, x, y) {
  im <- grayscale(load.image(image.files[[n]]))
  plot.array(im[x,y,,])
}


# Hour and minute
# x <- 2596:2775
# y <- 282:382

# First find only those where the hour exists.

# Load image files
image.files <- df$file.path
ids <- tools:::file_path_sans_ext(basename(image.files))
dir <- "/data/run_sheets_full"
files <- with(df, paste(dir, "R48_RIP", NoCd, rip, file, sep = "/"))

# Checkbox
x <- 2541:3881
y <- 1439:2600
# y <- 1150:2937

# Crop.
images <- mclapply(files, crop, x = x, y = y)

# Save raw images as arrays.
dir.create("checkboxes/data")
saveRDS(mclapply(images, to_2d_array), file = "checkboxes/data/raw_checkbox_images.rds")
# images <- readRDS("checkboxes/data/raw_checkbox_images.rds")

# Downsampling for all combinations of these parameters
scales <- c(0.4, 0.3, 0.2)
interp <- c(3, 1) # Linear and nearest

downscale.params <- expand.grid(scale = scales, interp = interp)

for (row in 1:nrow(downscale.params)) {
  scale <- downscale.params$scale[row]
  interp <- downscale.params$interp[row]
  
  arrays <- 
    mclapply(images, downsample, 
             scale = scale, interp = interp)
  
  file <- paste0("checkboxes_", "scale_", scale, "_interp_", interp, ".rds")
  
  saveRDS(arrays, file = paste0("checkboxes/data/", file))
}

write.array.to.image <- function(im, file) {
  if(length(dim(im)) == 2) dim(im) <- c(dim(im), 1, 1)
  save.image(cimg(im), file = file)
}

checkboxes_scale_0.2_interp_3 <- 
  readRDS("checkboxes/data/checkboxes_scale_0.2_interp_3.rds")

dir.create("little_images")
little.files <- paste0("little_images/", basename(files))
invisible(
  mapply(write.array.to.image, checkboxes_scale_0.2_interp_3, little.files)
)

