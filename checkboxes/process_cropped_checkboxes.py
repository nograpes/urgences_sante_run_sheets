import pandas as pd
import cv2
import pathlib
from collections import namedtuple
import pickle

data = pd.read_csv('checkboxes/cropped_checkboxes.csv')
files = data['file'].to_numpy()
checkbox = data['checked'].to_numpy()
paths = "checkboxes/cropped_checkboxes/" + files
images = [cv2.imread(str(path), cv2.COLOR_BGR2GRAY) for path in paths]
TimeImage = namedtuple("TimeImage", ["checkbox", "image", "path"])
checkbox_images_paths = [h for h in  map(TimeImage, checkbox, images, paths)]

with open('checkboxes/cropped_checkboxes.pkl', 'wb') as file:
  pickle.dump(checkbox_images_paths, file)

data = pd.read_csv('checkboxes/cropped_checkboxes_uncoded.csv')
data['checked'] = 0 # Set all to zero
files = data['file'].to_numpy()
checkbox = data['checked'].to_numpy()
paths = "checkboxes/cropped_checkboxes/" + files
images = [cv2.imread(str(path), cv2.COLOR_BGR2GRAY) for path in paths]
TimeImage = namedtuple("TimeImage", ["checkbox", "image", "path"])
checkbox_images_paths = [h for h in  map(TimeImage, checkbox, images, paths)]

with open('checkboxes/cropped_checkboxes_uncoded.pkl', 'wb') as file:
  pickle.dump(checkbox_images_paths, file)
