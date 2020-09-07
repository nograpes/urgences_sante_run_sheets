import os
import datetime
import shutil
import pathlib

# Gast 0.2.2 exactly is required. Gast 0.3 removes the 'Num'
# https://github.com/tensorflow/tensorflow/issues/32319
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.callbacks import TensorBoard

from keras import optimizers

import numpy as np
from random import sample, seed, shuffle
import math

import pickle
from collections import namedtuple
TimeImage = namedtuple("TimeImage", ["checkbox", "image", "path"])

# Run on unchecked
with open('checkboxes/cropped_checkboxes_uncoded.pkl', 'rb') as file:
  checkboxes_images_paths_unchecked = pickle.load(file)

unchecked_labels, unchecked_images, unchecked_paths = zip(* checkboxes_images_paths_unchecked)
unchecked_labels_dummy = keras.utils.to_categorical(unchecked_labels, 2)
unchecked_images = np.stack(unchecked_images) / 255.0
unchecked_images_4d = np.reshape(unchecked_images, unchecked_images.shape + tuple([1]))

import pandas as pd
data = pd.DataFrame(checkboxes_images_paths_unchecked)
data = data.drop(columns = 'image') 

data['file'] = list(map(os.path.basename, data['path']))
no_ext = [os.path.splitext(file)[0] for file in data['file']]
data['big_file'] = [base.split('_')[0] + ".png" for base in no_ext]
data['checkbox_name'] = [base.split('_')[1] for base in no_ext]


model = keras.models.load_model('checkboxes/checkbox_model')
predictions = np.array([np.argmax(prediction) for prediction in  model.predict(unchecked_images_4d)])

data['checkbox'] = predictions
data.to_csv("checkboxes/uncoded_predictions_long.csv", index = False)

import cv2
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
mypath = "aligned"

files = np.unique(data['big_file'])

for file in files:
    im = cv2.imread("/".join(("checkboxes/aligned", file)))
    for nm, tl in zip(names, top_left_corner):
        # Get checked by file nm
        is_checked = data['checkbox'].to_numpy()[(data['checkbox_name'] == nm) & (data['big_file'] == file)][0] == 1
        if is_checked:
            # Draw the box
            im = cv2.rectangle(
                   im, 
                   (tl[0], tl[1]), 
                   (tl[0] + square_size, tl[1] + square_size), 
                   (0, 255, 0), 
                   5
                 ) 
    im = im[1439:2600, 2541:3881]
    _ = cv2.imwrite("/".join(("checkboxes/uncoded_green_boxes", file)), im)

