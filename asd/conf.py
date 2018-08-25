""" Some useful constants.
"""

import os

BASE_DATA_PATH = os.path.dirname(os.path.realpath(__file__))
print(BASE_DATA_PATH)
MASKS_DATA_PATH = os.path.join(BASE_DATA_PATH, 'data', 'train_ship_segmentations.csv')
TRAIN_IMAGES_FOLDER = os.path.join(BASE_DATA_PATH, 'data', 'train')
VALID_SIZE = 0.2
FILE_SIZE_KB_THRESHOLD = 50
# For reproducibility
SEED = 42
# How many samples per ships bucket
SAMPLES_PER_SHIPS_BUCKET = 1500
