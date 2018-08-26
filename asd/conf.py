""" Some useful constants.
"""

import os

from hyperopt import hp

BASE_DATA_PATH = os.path.dirname(os.path.realpath(__file__))
MASKS_DATA_PATH = os.path.join(BASE_DATA_PATH, 'data', 'train_ship_segmentations.csv')
TRAIN_IMAGES_FOLDER = os.path.join(BASE_DATA_PATH, 'data', 'train')
TEST_IMAGES_FOLDER = os.path.join(BASE_DATA_PATH, 'test')
VALID_SIZE = 0.2
FILE_SIZE_KB_THRESHOLD = 50
# For reproducibility
SEED = 42
# How many samples per ships buckets
# TODO: Should this be a hyperparamter?
# TODO: Find better ways to rebalance the data.
SAMPLES_PER_SHIPS_BUCKET = 2000


# Global constants
SEED = 31415
# Set it to a small number so that it can run in this notebook.
# Notice that if it is too small, hyperopt behaves as random selection.
MAX_EVALS = 1
CUSTOM_DICE_LOSS_EPSILON = 1e-3
# According to the data description, some files from the test folder shoud be ignore
TEST_IMGS_TO_IGNORE = ['13703f040.jpg',
                       '14715c06d.jpg',
                       '33e0ff2d5.jpg',
                       '4d4e09f2a.jpg',
                       '877691df8.jpg',
                       '8b909bb20.jpg',
                       'a8d99130e.jpg',
                       'ad55c3143.jpg',
                       'c8260c541.jpg',
                       'd6c7f17c7.jpg',
                       'dc3e7c901.jpg',
                       'e44dffe88.jpg',
                       'ef87bad36.jpg',
                       'f083256d8.jpg']
# These two patiences thresholds are small so that this notebook can run with limited resources
REDUCE_LR_PATIENCE = 2
EARLY_STOPPING_PATIENCE = 2
# Fraction of the validation size (compared to the total train size)
VALID_SIZE = 0.3
# Minimum size (in KB) of files to keep
FILE_SIZE_KB_THRESHOLD = 50
# The original size of the image
# TODO: Check if it is really 3 channels.
IMG_SIZE = (768, 768, 3)
# downsampling in preprocessing
# TODO: Should these be hp to optimize as well?
IMG_SCALING = (4, 4)
EDGE_CROP = 16
# downsampling inside the network
NET_SCALING = (1, 1)
# number of validation images to use
VALID_IMG_COUNT = 600
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 150
MAX_TRAIN_EPOCHS = 10
# The hyperparameters space over which to search.
# TODO: Improve the ranges and the used distributions to sample.
HYPERPARAMETERS_SPACE = {
    # TODO: What is the best scale for Gaussian noise?
    'gaussian_noise': hp.choice('gaussian_noise', [0.1, 0.2, 0.3]),
    'batch_size':  hp.choice('batch_size', [8, 16, 32, 64, 128]),
    'upsample_mode': hp.choice('upsmaple_mode', ["SIMPLE", "DECONV"]),
    'augment_brightness': hp.choice('augment_brightness', [True, False]),
    'max_train_steps': MAX_TRAIN_STEPS,
    'max_train_epochs': MAX_TRAIN_EPOCHS,
    'valid_img_count': VALID_IMG_COUNT,
    'img_scaling': IMG_SCALING,
    'edge_crop': EDGE_CROP,
    'net_scaling': NET_SCALING
}
