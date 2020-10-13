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
from pathlib import Path
import multiprocessing as mp

import pickle
import copyreg


MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.40
PERC_COMMON_THRESH = 0.40
KEYPOINT_COLOR_MATCH_FILE = (20, 200, 20)
MATCH_FILE_NAME = "matches.jpg"
COMPOSITE_FILE_NAME = "composite.jpg"

ref_filename = "/data/run_sheets_full/R48_RIP/E441/RIP3/21543780.png"
out_dir = "/data/run_sheets_full/aligned/"

# Read all the file_df data in.
file_df = pd.read_csv("checkboxes/file_df.csv", low_memory = False)
file_df['aligned_path'] = [img.replace('/data/run_sheets_full/', out_dir) 
                           for img in file_df['file.path']]
sheets = file_df['sheet_type'] == 'rev 2017-05'
imgs = np.array(file_df['file.path'][sheets])
out_imgs = np.array(file_df['aligned_path'][sheets])

out_dirs = set(map(os.path.dirname, map(os.path.abspath, out_imgs)))
for out_dir in out_dirs: Path(out_dir).mkdir(parents=True, exist_ok=True)

compositor = Compositor.get_additive()

def prepare_img(filename):
    return cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

def find_matches(orb, img, ref_descriptors, match_quality_threshold):
    # Detect ORB features and compute descriptors.
    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors, ref_descriptors, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Remove not so good matches
    num_match_to_keep = int(len(matches) * match_quality_threshold)
    matches = matches[:num_match_to_keep]
    
    return keypoints, matches

# Patch the DMatch and KeyPoint objects in CV2 so that they are picklable.
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

def _pickle_dmatches(point):
    return cv2.DMatch, (point.queryIdx, point.trainIdx, point.imgIdx, point.distance)

copyreg.pickle(cv2.DMatch().__class__, _pickle_dmatches)

orb = cv2.ORB_create(MAX_FEATURES)

# Load Reference Image
ref = prepare_img(ref_filename)
ref_keypoints, ref_descriptors = orb.detectAndCompute(ref, None)

def find_match_file(file):
    return find_matches(orb, prepare_img(file), ref_descriptors, GOOD_MATCH_PERCENT)

def find_match_img(img):
    return find_matches(orb, img, ref_descriptors, GOOD_MATCH_PERCENT)

# Hyperthreading seems to speed things up.
pool = mp.Pool(mp.cpu_count())
short_imgs = imgs[:5000]

start = timer()
keypoints, matches = zip(*pool.map(find_match_file, (short_imgs)))
end = timer()
print("Processed matches in %s sec" % round(end - start, 2))

keypoints_dic = dict(zip(short_imgs, keypoints))
matches_dic = dict(zip(short_imgs, matches))

# A before composite is difficult to do here, because the images don't all have the same size

# Count the usage for each descriptor index.
common_pts = {}
for filename, matches in matches_dic.items():
    local_common = {}
    for match in matches:
        # increase view count for the index of this key point in the reference.
        local_common[match.trainIdx] = local_common.get(match.trainIdx, 0) + 1
    for idx, num_uses in local_common.items():
        if num_uses != 1:  # prevent descriptors with multiple matches to be counted.
            continue
        common_pts[idx] = common_pts.get(idx, 0) + 1


# find the commonest of common point (can't inverse dict because of non-unique lines.)
# Keep only points for which matches were found in at least 30% of the images.
threshold = PERC_COMMON_THRESH * len(short_imgs)
idx_to_use = []
for idx, num_uses in common_pts.items():
    if num_uses < threshold:
        continue
    else:
        idx_to_use.append(idx)

to_draw = [kpt for i, kpt in enumerate(ref_keypoints) if i in idx_to_use]
img2 = cv2.drawKeypoints(ref, to_draw, None, color=KEYPOINT_COLOR_MATCH_FILE, flags=0)
cv2.imwrite(MATCH_FILE_NAME, img2)

def warp_img_by_matches(img, ref, ref_keypoints, keypoints, matches, idx_to_use):
    matches_to_use = [m for m in matches if m.trainIdx in idx_to_use]
    
    # Extract location of good matches
    points1 = np.zeros((len(matches_to_use), 2), dtype=np.float32)
    points2 = np.zeros((len(matches_to_use), 2), dtype=np.float32)
    
    for i, match in enumerate(matches_to_use):
        points1[i, :] = keypoints[match.queryIdx].pt
        points2[i, :] = ref_keypoints[match.trainIdx].pt

    # Use homography
    height, width = ref.shape
    
    # Wrap and return the data
    return cv2.warpPerspective(img, h, (width, height))


warp_img_by_matches(short_imgs[0], ref, ref_keypoints, keypoints, matches, idx_to_use)

filename = short_imgs[0]
out_img = out_imgs[0]




img = prepare_img(filename)
dat, _ = warp_img_by_matches(
    img, ref, ref_keypoints, keypoints_dic[filename], matches_dic[filename], idx_to_use)

new_path = "%s/%s" % (out_dir, os.path.basename(filename))
cv2.imwrite(new_path, dat)

print("Starting warping")
start = timer()
#  composite = Image.fromarray(ref)
for filename, new_path in (short_imgs, out_imgs[:5000]):
    img = prepare_img(filename)
    # Warp image
    dat, _ = warp_img_by_matches(
        img, ref, ref_keypoints, keypoints_dic[filename], matches_dic[filename], idx_to_use)
    
    # Write to file
    cv2.imwrite(new_path, dat)
