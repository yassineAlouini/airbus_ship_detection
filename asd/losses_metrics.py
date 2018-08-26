""" Some useful losses and metrics for instance segmentation and unbalanced image classification
"""


import keras.backend as K

from asb.conf import CUSTOM_DICE_LOSS_EPSILON


def dice_metric(y_true, y_pred, smooth=1):
    """
    Also known as the Sorensen-Dice coeffecient (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient),
    this is the F1 score (i.e. harmonic mean of precision and recall).
    Notice that this metric has a smoothness parameter (smooth) to avoid division by 0.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    # Compute the dice metric and then take the average over the samples.
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def custom_dice_loss(in_gt, in_pred):
    """ This is a custom loss function that has two contributions: binary crossentropy
    (this is the usual metric used for binary classification) and - the dice metric (to turn it into a loss).
    """
    return CUSTOM_DICE_LOSS_EPSILON * binary_crossentropy(in_gt, in_pred) - dice_metric(in_gt, in_pred)


def true_positive_rate_metric(y_true, y_pred):
    """ TPR (true positive rate) measures the ratio of true positives over positives.
    Notice the round step so that the predicted values are transformed into 0 or 1 values instead of floats
    in the range [0, 1].
    """
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)


METRICS = [true_positive_rate_metric,
           dice_metric,
           "binary_accuracy"]
