""" Some useful losses and metrics for instance segmentation and unbalanced image classification
"""


import keras.backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy

from asd.conf import CUSTOM_DICE_LOSS_EPSILON, CUSTOM_FOCAL_LOSS_EPSILON


def dice_metric(y_true, y_pred, smooth=1.0):
    """
    Also known as the Sorensen-Dice coeffecient (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient),
    this is the F1 score (i.e. harmonic mean of precision and recall).
    Notice that this metric has a smoothness parameter (smooth) to avoid division by 0.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    # Compute the dice metric and then take the average over the samples.
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def IoU_metric(y_true, y_pred, smooth=1.0):
    """
    Also known as the Sorensen-Dice coeffecient (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient),
    this is the F1 score (i.e. harmonic mean of precision and recall).
    Notice that this metric has a smoothness parameter (smooth) to avoid division by 0.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    # Compute the dice metric and then take the average over the samples.
    return K.mean(intersection / (union + smooth), axis=0)

# From here: https://github.com/mkocabas/focal-loss-keras/blob/master/focal_loss.py
# TODO: Make this agnostic of tensorflow.


def focal_loss(gamma=2., alpha=1.0):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

# Inspired from here: https://www.kaggle.com/iafoss/unet34-dice-0-87#


def custom_focal_loss(y_true, y_pred):
    # TODO: Add some documentation
    return focal_loss()(y_true, y_pred) - 0.25 * K.log(dice_metric(y_true, y_pred))


def custom_dice_loss(y_true, y_pred):
    """ This is a custom loss function that has two contributions: binary crossentropy
    (this is the usual metric used for binary classification) and - the dice metric (to turn it into a loss).
    """
    return binary_crossentropy(y_true, y_pred) + 0.25 * (1 - dice_metric(y_true, y_pred))


def true_positive_rate_metric(y_true, y_pred):
    """ TPR (true positive rate) measures the ratio of true positives over positives.
    Notice the round step so that the predicted values are transformed into 0 or 1 values instead of floats
    in the range [0, 1].
    """
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)


METRICS = [true_positive_rate_metric, IoU_metric, dice_metric, "binary_accuracy"]
