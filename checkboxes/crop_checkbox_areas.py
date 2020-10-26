import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from os.path import basename

# Checkbox crop
tl = [2580, 1500]

top_left = pd.read_csv("checkboxes/top_left.csv")
top_left['x'] = top_left['x'].to_numpy() - tl[0]
top_left['y'] = top_left['y'].to_numpy() - tl[1]

top_left_dict = {}
for key, val in top_left.groupby('checkbox_layout'):
    coords = list(zip(val['x'], val['y']))
    names = val['checkbox']
    top_left_dict[key] = dict(zip(names, coords))

square_size = 27

root = "/data"
out_dir = root + "/run_sheets_full/checkbox_area_aligned"

# Read all the file_df data in.
file_df = pd.read_csv("checkboxes/file_df_checkbox_area.csv", low_memory = False)
layout_df = pd.read_csv("print_run_to_sheet_type.csv", low_memory = False)
layouts = dict(zip(layout_df['print_run'], layout_df['checkbox_layout']))

print_runs = np.unique(file_df['print_run'])
print_runs = print_runs[print_runs != 'impr 2011-01']

# Greenbox sample
# sampled_images = dict()
# np.random.seed(1)
# for print_run in print_runs:
#     rows = file_df['print_run'] == print_run
#     imgs = file_df['checkbox_area'][rows]
#     imgs = \
#       [img.replace('checkbox_area', 'checkbox_area_aligned') \
#        for img in imgs]
#     sampled = np.random.choice(np.array(imgs), 5)
#     sampled_images[print_run] = sampled
# 
# 
# Path("checkboxes/greenbox_samples").mkdir(parents=True, exist_ok=True)
# 
# for print_run in sampled_images:
#     layout = layouts[print_run]
#     top_left_corner = top_left_dict[layout]
#     for file in sampled_images[print_run]:
#         im = cv2.imread(file)
#         for name, tl in top_left_corner.items():
#             im = cv2.rectangle(
#                    im, 
#                    (tl[0], tl[1]), 
#                    (tl[0] + square_size, tl[1] + square_size), 
#                    (0, 255, 0), 
#                    5
#                  ) 
#         out_file = "checkboxes/greenbox_samples/" + print_run + "_" + basename(file)
#         _ = cv2.imwrite(out_file, im)


# Cutting
sheets = file_df['sheet_type'] != 'impr 2011-01'
print_runs = np.array(file_df['print_run'][sheets])
imgs = np.array(file_df['checkbox_area'][sheets])
imgs = \
  [img.replace("checkbox_area", "checkbox_area_aligned") \
  for img in imgs]

out_imgs_prefix = \
  [img.replace("checkbox_area_aligned", "checkbox_cropped").replace(".png", "") \
  for img in imgs]

from os import listdir
from os.path import isfile, join, splitext, dirname, abspath
out_dirs = set(map(dirname, map(abspath, out_imgs)))
for out_dir in out_dirs: 
    Path(out_dir).mkdir(parents=True, exist_ok=True)


for print_run, file, out_img_prefix in tqdm(zip(print_runs, imgs, out_imgs_prefix)):
    layout = layouts[print_run]
    top_left_corner = top_left_dict[layout]
    im = cv2.imread(file)
    for name, tl in top_left_corner.items():
        br = (tl[0] + square_size, tl[1] + square_size)
        new_im = im[tl[1]:br[1], (tl[0]+1):(br[0]+1)]
        out_file = out_img_prefix + "_" + name + ".png"
        _ = cv2.imwrite(out_file, new_im)

