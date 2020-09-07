import cv2
import pandas as pd
top_left = pd.read_csv("checkboxes/top_left.csv")

x = top_left['x'].to_numpy()
y = top_left['y'].to_numpy()

square_size = 27

names = top_left['checkbox'].to_numpy()
top_left_corner = \
  list(zip(x, y))
bottom_right_corner = \
  list(zip(x + square_size, y + square_size))


from os import listdir
from os.path import isfile, join, splitext
mypath = "checkboxes/aligned"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in files:
    basename = splitext(file)[0]
    im = cv2.imread("/".join(("aligned", file)), cv2.COLOR_BGR2GRAY)
    for nm, tl, br in zip(names, top_left_corner, bottom_right_corner):
        new_im = im[tl[1]:br[1], (tl[0]+1):(br[0]+1)]
        newname = basename + "_" + nm + ".png"
        _ = cv2.imwrite("/".join(("checkboxes/cropped_checkboxes", newname)), new_im)


