import os
import datetime
import shutil
import pathlib

# Gast 0.2.2 exactly is required. Gast 0.3 removes the 'Num'
# https://github.com/tensorflow/tensorflow/issues/32319
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

with open('checkboxes/cropped_checkboxes.pkl', 'rb') as file:
  checkboxes_images_paths = pickle.load(file)

# Only the first 1000 were validated by me.
#checkboxes_images_paths = checkboxes_images_paths[:1000]

test_prop = 0.2
seed(1)
shuffle(checkboxes_images_paths)
test_num = math.floor(len(checkboxes_images_paths) * test_prop)

test_labels, test_images, test_paths = zip(*checkboxes_images_paths[0:test_num])
train_labels, train_images, train_paths = zip(*checkboxes_images_paths[test_num+1:])

train_labels_dummy = keras.utils.to_categorical(train_labels, 2)
test_labels_dummy = keras.utils.to_categorical(test_labels, 2)

test_images = np.stack(test_images) / 255.0
train_images = np.stack(train_images) / 255.0

input_shape = (27,27)
num_classes = 2

train_images_4d = np.reshape(train_images, train_images.shape + tuple([1]))
test_images_4d = np.reshape(test_images, test_images.shape + tuple([1]))

model = Sequential()
model.add(Conv2D(64, (9, 9), activation='relu'))
model.add(MaxPooling2D(pool_size=(9, 9)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1280, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

log_dir = "checkboxes/logs"
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
#tensorboard = TensorBoard(log_dir = log_dir)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = adam, # tf.train.AdamOptimizer(),
	loss = keras.losses.categorical_crossentropy,
	metrics=["accuracy"],
	)

model.fit(x=train_images_4d, 
          y=train_labels_dummy, 
          verbose = 0, # 0 for silent, 1 for normal.
          epochs=50, 
          validation_data=(test_images_4d, test_labels_dummy), 
          #callbacks=[tensorboard]
          )
model.save('checkboxes/checkbox_model')


"""
score = model.evaluate(test_images_4d, test_labels_dummy, verbose=0)

predictions = np.array([np.argmax(prediction) for prediction in  model.predict(test_images_4d)])
test_label_nums = list(map(int, test_labels))
missed_files = np.array(test_paths)[predictions != test_label_nums]
missed_nums = np.array(test_label_nums)[predictions != test_label_nums]
not_missed_nums = np.array(test_label_nums)[predictions == test_label_nums]

# Among the test set which checked did you get wrong?
missed_num1 = \
  np.array(test_paths)[(predictions != test_label_nums) & 
  (np.array(test_label_nums) == 1)]


predictions = np.array([np.argmax(prediction) for prediction in  model.predict(train_images_4d)])
train_label_nums = list(map(int, train_labels))
missed_files = np.array(train_paths)[predictions != train_label_nums]
missed_nums = np.array(train_label_nums)[predictions != train_label_nums]
not_missed_nums = np.array(train_label_nums)[predictions == train_label_nums]
missed_num1 = np.array(train_paths)[(predictions != train_label_nums) & (np.array(train_label_nums) == 0)]

for file in missed_num1: shutil.copy(file, "errors")





misses_by_num = np.unique(missed_nums, return_counts = True)[1]
total_by_num = np.unique(test_label_nums, return_counts = True)[1]

directory = missed_files[0].parent.parent / 'missed'
shutil.rmtree(directory, ignore_errors = True)
pathlib.Path.mkdir(directory)

for file in missed_files:
  dest = file.parent.parent / 'missed' / file.name
  _ = shutil.copy(file, dest)
"""

