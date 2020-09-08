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
import multiprocessing as mp

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.40
PERC_COMMON_THRESH = 0.40
KEYPOINT_COLOR_MATCH_FILE = (20, 200, 20)
MATCH_FILE_NAME = "matches.jpg"
COMPOSITE_FILE_NAME = "composite.jpg"

ref_filename = "/data/run_sheets_full/R48_RIP/E441/RIP3/21543780.png"
out_dir = "/data/run_sheets_full/aligned/2017-05"

# Read all the file_df data in.
file_df = pd.read_csv("checkboxes/file_df.csv", low_memory = False)
imgs = np.array(file_df['file.path'][file_df['sheet_type'] == 'rev 2017-05'])

compositor = Compositor.get_additive()
# aligner = Align(ref_filename, imgs, out_dir, compositor)
# aligned_dic = aligner.align()


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


orb = cv2.ORB_create(MAX_FEATURES)

# Load Reference Image
ref = prepare_img(ref_filename)
ref_keypoints, ref_descriptors = orb.detectAndCompute(ref, None)

def find_match_file(file):
    return find_matches(orb, prepare_img(file), ref_descriptors, GOOD_MATCH_PERCENT)

def find_match_img(img):
    return find_matches(orb, img, ref_descriptors, GOOD_MATCH_PERCENT)


import pickle
import copyreg
import cv2

def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)


def _pickle_dmatches(point):
    return cv2.DMatch, (point.queryIdx, point.trainIdx, point.imgIdx, point.distance)

copyreg.pickle(cv2.DMatch().__class__, _pickle_dmatches)

pool = mp.Pool(mp.cpu_count())
# mem_imgs = pool.map(prepare_img, (imgs[:500]))

short_imgs = imgs[:5000]

start = timer()
keypoints, matches = zip(*pool.map(find_match_file, (short_imgs)))
end = timer()
print("Processed matches in %s sec" % round(end - start, 2))

keypoints_dic = dict(zip(short_imgs, keypoints))
matches_dic = dict(zip(short_imgs, matches))

results = {}

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
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Use homography
    height, width = ref.shape
    
    # Wrap and return the data
    return cv2.warpPerspective(img, h, (width, height)), h

filename = short_imgs[0]
img = prepare_img(filename)
dat, hom = warp_img_by_matches(
    img, ref, ref_keypoints, keypoints_dic[filename], matches_dic[filename], idx_to_use)
new_path = "%s/%s" % (out_dir, os.path.basename(filename))
cv2.imwrite(new_path, dat)



print("Starting warping")
start = timer()
#  composite = Image.fromarray(ref)
for i, filename in enumerate(tqdm(short_imgs)):
    img = prepare_img(filename)
    # Warp image
    dat, hom = warp_img_by_matches(
        img, ref, ref_keypoints, keypoints_dic[filename], matches_dic[filename], idx_to_use)
    
    # Add to composite
    # self.compositor.compose(dat)
    #  composite = Image.blend(composite, Image.fromarray(dat), alpha=1/(i+1))
    #  composite = add_composite(composite, dat)
    
    # Write to file
    new_path = "%s/%s" % (self.out_dir, os.path.basename(filename))
    cv2.imwrite(new_path, dat)
