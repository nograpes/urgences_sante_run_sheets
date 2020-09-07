import pandas as pd
import numpy as np

from img_align.Align import Align
from img_align.Compositor import Compositor

import cv2
import numpy as np
from timeit import default_timer as timer
from PIL import Image
from tqdm import tqdm
import os

ref_filename = "checkboxes/big_images/21543780.png"
out_dir = "D:\\urgence_sante\\aligned\\2017-05"

# Read all the file_df data in.
file_df = pd.read_csv("checkboxes/file_df.csv", low_memory = False)
imgs = np.array(file_df['file.path'][file_df['sheet_type'] == 'rev 2017-05'])

compositor = Compositor.get_additive()
aligner = Align(ref_filename, imgs, out_dir, compositor)
aligned_dic = aligner.align()

