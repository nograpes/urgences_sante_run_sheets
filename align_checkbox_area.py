import pandas as pd
import numpy as np

# from img_align.Align import Align

import cv2
import numpy as np
from timeit import default_timer as timer
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map, cpu_count 
import os
from pathlib import Path

import pickle
import copyreg

from functools import partial

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.40
PERC_COMMON_THRESH = 0.40
KEYPOINT_COLOR_MATCH_FILE = (20, 200, 20)
MATCH_FILE_NAME = "matches.jpg"
orb = cv2.ORB_create(MAX_FEATURES)


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

def find_matches2(file, max_features, ref_descriptors, match_quality_threshold):
    img = prepare_img(file)
    orb = cv2.ORB_create(max_features)
    
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
# Also the ORB object.
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

def _pickle_dmatches(point):
    return cv2.DMatch, (point.queryIdx, point.trainIdx, point.imgIdx, point.distance)

copyreg.pickle(cv2.DMatch().__class__, _pickle_dmatches)

def _pickle_ORB(point):
    return cv2.ORB, (point.queryIdx, point.trainIdx, point.imgIdx, point.distance)

copyreg.pickle(cv2.ORB().__class__, _pickle_ORB)


def find_match_file(file):
    return find_matches(orb, prepare_img(file), ref_descriptors, GOOD_MATCH_PERCENT)

def find_match_img(img):
    return find_matches(orb, img, ref_descriptors, GOOD_MATCH_PERCENT)

def warp_img_by_matches(img, ref, ref_keypoints, keypoints, matches, idx_to_use):
    matches_to_use = [m for m in matches if m.trainIdx in idx_to_use]
    
    # Extract location of good matches
    points1 = np.zeros((len(matches_to_use), 2), dtype=np.float32)
    points2 = np.zeros((len(matches_to_use), 2), dtype=np.float32)
    
    for i, match in enumerate(matches_to_use):
        points1[i, :] = keypoints[match.queryIdx].pt
        points2[i, :] = ref_keypoints[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Use homography
    height, width = ref.shape
    
    # Warp and return the data
    return cv2.warpPerspective(img, h, (width, height))

def warp_img(filename, out_img, keypoints, matches):
    img = prepare_img(filename)
    dat = warp_img_by_matches(
           img, ref, ref_keypoints, 
           keypoints, matches, 
           idx_to_use)
    _ = cv2.imwrite(out_img, dat)


def warp_img2(tup, ref, ref_keypoints, idx_to_use):
    file, out_img, keypoints, matches = tup
    img = prepare_img(file)
    dat = warp_img_by_matches(
           img, ref, ref_keypoints, 
           keypoints, matches, 
           idx_to_use)
    _ = cv2.imwrite(out_img, dat)


def align_images(infiles, outfiles, ref, ref_keypoints, ref_descriptors):
    short_imgs = infiles    
    
    find_match_img2 = \
       partial(find_matches2, 
               max_features = MAX_FEATURES, 
               ref_descriptors = ref_descriptors, 
               match_quality_threshold = GOOD_MATCH_PERCENT)
    # print("Starting matching")
    start = timer()
    keypoints, matches = zip(*thread_map(find_match_img2, (short_imgs), max_workers = cpu_count() // 2, position = 1))
    end = timer()
    # print("Processed matches in %s sec" % round(end - start, 2))
    
    # Count the usage for each descriptor index.
    common_pts = {}
    for filename, matches1 in zip(short_imgs, matches):
        local_common = {}
        for match in matches1:
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
    
    warp_img_p = partial(warp_img2, ref = ref, ref_keypoints = ref_keypoints, idx_to_use = idx_to_use)
    # print("Starting warping")
    start = timer()
    _ = thread_map(warp_img_p, list(zip(short_imgs, outfiles, keypoints, matches)), max_workers = cpu_count() // 2, position = 1)
    end = timer()
    # print("Processed warping in %s sec" % round(end - start, 2))


# root = "D:/urgence_sante"
root = "/data"

out_dir = root + "/run_sheets_full/checkbox_area_aligned"

# Read all the file_df data in.
file_df = pd.read_csv("checkboxes/file_df_checkbox_area.csv", low_memory = False)

sheets = file_df['sheet_type'] == 'rev 2017-05'
imgs = np.array(file_df['checkbox_area'][sheets])

out_imgs = \
  [img.replace("checkbox_area", "checkbox_area_aligned") \
  for img in imgs]

out_dirs = set(map(os.path.dirname, map(os.path.abspath, out_imgs)))
for out_dir in out_dirs: Path(out_dir).mkdir(parents=True, exist_ok=True)

# Desired interface. 
# A function that I feed a bunch of in files and out files.
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

chunksize = 1000
ref_filename = imgs[0]
# Load Reference Image
orb = cv2.ORB_create(MAX_FEATURES)
ref = prepare_img(ref_filename)
ref_keypoints, ref_descriptors = orb.detectAndCompute(ref, None)

for img_chunk, out_img_chunk in tqdm(zip(chunks(imgs, chunksize), chunks(out_imgs, chunksize)), position = 0):
    align_images(img_chunk, out_img_chunk, ref, ref_keypoints, ref_descriptors)


sheets = file_df['sheet_type'] == 'rev 2012-01'
imgs = np.array(file_df['checkbox_area'][sheets])


out_imgs = \
  [img.replace("checkbox_area", "checkbox_area_aligned") \
  for img in imgs]

out_dirs = set(map(os.path.dirname, map(os.path.abspath, out_imgs)))
for out_dir in out_dirs: Path(out_dir).mkdir(parents=True, exist_ok=True)

# Desired interface. 
# A function that I feed a bunch of in files and out files.
ref_filename = imgs[0]
# Load Reference Image
orb = cv2.ORB_create(MAX_FEATURES)
ref = prepare_img(ref_filename)
ref_keypoints, ref_descriptors = orb.detectAndCompute(ref, None)

for img_chunk, out_img_chunk in zip(chunks(imgs, chunksize), chunks(out_imgs, chunksize)):
    align_images(img_chunk, out_img_chunk, ref, ref_keypoints, ref_descriptors)

