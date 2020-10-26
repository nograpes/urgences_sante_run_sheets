import cv2
import pandas as pd
import numpy as np
from os.path import dirname
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer
import multiprocessing as mp


# root = "D:/urgence_sante"
root = "/data"
out_dir = root + "/run_sheets_full/checkbox_area/"

# Read all the file_df data in.
file_df = pd.read_csv("checkboxes/file_df.csv", low_memory = False)
file_df['file.path'] = \
    [img.replace('/data', root) for img in file_df['file.path']]

file_df['checkbox_area'] = \
    [img.replace(root + "/run_sheets_full/", out_dir) 
    for img in file_df['file.path']]

files = np.array(file_df['file.path'])
out_files = np.array(file_df['checkbox_area'])

file_df.to_csv("checkboxes/file_df_checkbox_area.csv")

tl = [2580, 1500]
br = [3837, 2961]

dirs = np.unique(list(map(dirname, out_files)))
for dir in dirs:
    Path(dir).mkdir(parents=True, exist_ok=True)    

def crop(file, out_file):
    im = cv2.imread(file, cv2.COLOR_BGR2GRAY)
    new_im = im[tl[1]:br[1], tl[0]:br[0]]
    cv2.imwrite(out_file, new_im)

pool = mp.Pool(48)
start = timer()
_ = pool.starmap(crop, zip(files, out_files))
end = timer()
print("Processed in %s sec" % round(end - start, 2))
