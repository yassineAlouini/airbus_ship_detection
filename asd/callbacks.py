""" Model callbacks setup.
"""

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from asd.conf import EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE

# TODO: Add some documentation for these callbacks.
# TODO: Import the FbetaMetricCallback from kaggle_tools package.

weight_path = "best_weights.h5"

# TODO: Move some of the hyperparameters to the constants list
MODEL_CHECKPOINT_CALLABACK = ModelCheckpoint(weight_path, monitor='val_loss',
                                             verbose=1, save_best_only=True,
                                             mode='min', save_weights_only=True)

REDUCE_LR_CALLBACK = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=REDUCE_LR_PATIENCE, verbose=1, mode='min',
                                       min_delta=0.0001, cooldown=2, min_lr=1e-7)
# probably needs to be more patient, but kaggle time is limited
EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                                        patience=EARLY_STOPPING_PATIENCE)

# TODO: Add the FbetaMetricCallback
CALLBACKS = [MODEL_CHECKPOINT_CALLABACK, EARLY_STOPPING_CALLBACK, REDUCE_LR_CALLBACK]
