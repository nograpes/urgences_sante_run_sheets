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

# Checkbox
x <- 2541:3881
y <- 1439:2600
# y <- 1150:2937

# Crop.
system.time(images <- mclapply(image.files[1:2], crop, x = x, y = y))


# Save raw images as arrays.
saveRDS(mclapply(images, to_2d_array), file = "data/raw_hour_images.rds")

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
  
  saveRDS(arrays, file = paste0("data/", file))
}