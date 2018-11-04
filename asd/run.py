import gc
import glob
import os

import click
import numpy as np
import pandas as pd
from keras import models
from keras.optimizers import Adam
from skimage.io import imread
from skimage.morphology import binary_opening, disk
from tqdm import tqdm
from comet_ml import Experiment

from asd.callbacks import CALLBACKS
from asd.conf import (BEST_MODEL_PATH, EDGE_CROP, IMG_SCALING, IMG_SIZE,
                      MAX_TRAIN_EPOCHS, MAX_TRAIN_STEPS, NET_SCALING,
                      TEST_IMAGES_FOLDER, TEST_IMGS_TO_IGNORE, COMET_ML_API_KEY, 
                      PROJECT_NAME)
from asd.losses_metrics import (METRICS, custom_focal_loss, dice_metric,
                                true_positive_rate_metric)
from asd.models.pretrained_unet import build_pretrained_unet_model
from asd.models.u_net import build_u_net_model
from asd.preprocessing import (create_aug_gen, get_data, make_image_gen,
                               multi_rle_encode)

gc.enable()

# Comet experiement
experiment = Experiment(api_key=COMET_ML_API_KEY,
                        project_name=PROJECT_NAME,
                        auto_param_logging=False)

# TODO: Add some documentation


def get_compiled_model(hyperparameters, input_shape=IMG_SIZE, load_pretrained=True):
    if load_pretrained:
        # TODO: Add hyperparamters to this model later.
        # What about input_shape resizing?
        model = build_pretrained_unet_model()
        print("Using pretrained Unet model")
    else:
        model = build_u_net_model(input_shape, **hyperparameters)
    # TODO: These should be in the hp list as well.
    learning_rate = 1e-2
    decay = 1e-7
    adam_optimizer = Adam(learning_rate, decay=decay)
    model.compile(optimizer=adam_optimizer, loss=custom_focal_loss, metrics=METRICS)
    print(model.summary())
    return model


# TODO: Add some documentation.
# TODO: Use logger instead of print.
def ml_pipeline(input_train_df, input_valid_df, hyperparameters, n_samples, input_shape):
    # Copy input DataFrames to avoid side-effects
    train_df = input_train_df.copy()
    valid_df = input_valid_df.copy()
    # TODO: Improve the hp parsing section
    max_train_steps = hyperparameters["max_train_steps"]
    batch_size = hyperparameters["batch_size"]
    img_scaling = hyperparameters["img_scaling"]
    max_train_epochs = hyperparameters["max_train_epochs"]
    augment_brightness = hyperparameters["augment_brightness"]
    steps_per_epoch = min(max_train_steps, n_samples // batch_size)
    print("Using {} steps per epoch.".format(steps_per_epoch))
    img_genarator = make_image_gen(train_df, batch_size, img_scaling)
    # TODO:  Try this library for data augmentation
    # https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.PadIfNeeded
    augmented_img_generator = create_aug_gen(img_genarator, augment_brightness)
    # TODO: Improve the names of these returned values.
    # TODO: Replace this with a bigger validation set.
    valid_x, valid_y = next(make_image_gen(valid_df, batch_size, img_scaling))
    model = get_compiled_model(hyperparameters, input_shape)
    # Use only one worker for thread-safety reason.
    # TODO: Investigate this claim.
    history = model.fit_generator(augmented_img_generator, steps_per_epoch=steps_per_epoch,
                                  epochs=max_train_epochs, validation_data=(valid_x, valid_y),
                                  callbacks=CALLBACKS, workers=1)
    return {"history": history.history, "model": model}


def _predict_for_batch(img_paths, model, output_path):
    encoded_predictions = []
    img_ids = []
    for img_path in img_paths:
        img_id = img_path.split('/')[-1].replace('.jpg', '')
        if img_id not in TEST_IMGS_TO_IGNORE:
            img = imread(img_path)
            img = np.expand_dims(img, 0) / 255.0
            predictions = model.predict(img)[0]
            predictions = binary_opening(predictions > 0.5, np.expand_dims(disk(2), -1))
            encoded_rles = multi_rle_encode(predictions)
            if len(encoded_rles) > 0:
                for encoded_rle in encoded_rles:
                    encoded_predictions.append(encoded_rle)
                    img_ids.append(img_id)
            else:
                # No ship has been found
                encoded_predictions.append(None)
                img_ids.append(img_id)
    gc.collect()
    df = pd.DataFrame({"ImageId": img_ids, "EncodedPixels": encoded_predictions})
    if not os.path.isfile(output_path):
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, mode="a", header=False)


def prepare_submission(debug, model_path, output_path):
    """ Load the best trained models and predict the masks for the test images.
    """
    # Loda the model
    model = models.load_model(model_path, custom_objects={
                              'custom_dice_loss': custom_dice_loss,
                              'true_positive_rate_metric': true_positive_rate_metric,
                              'dice_metric': dice_metric})
    # TODO: Use batch of images to predict at once (otherwise too slow).
    if debug:
        test_files = glob.glob(os.path.join(TEST_IMAGES_FOLDER, "*.jpg"))[:10]
    else:
        test_files = glob.glob(os.path.join(TEST_IMAGES_FOLDER, "*.jpg"))
    for i in tqdm(range(0, len(test_files), 100)):
        img_paths = test_files[i:i + 100]
        _predict_for_batch(img_paths, model, output_path)
    return model


@click.command()
@click.option('--debug', type=bool,
              default=True, help='Whether to run the pipeline in debug mode or not. Defaults to True.')
@click.option('--train', type=bool,
              default=False, help='Whether to train a model or load one. Defaults to False.')
@click.option('--output_path', type=str, help='Where to store the submission file.')
def main(debug, train, output_path):
    if train:
        with experiment.train():
            train_df, valid_df = get_data()
            n_samples = train_df.shape[0]
            print(n_samples)
            input_shape = IMG_SIZE
            # Default hyperparameters.
            hyperparameters = {'gaussian_noise': 0.1,
                           'batch_size':  32, # Try lower if necessary.
                           'upsample_mode': "DECONV",
                           'augment_brightness': True,
                           'max_train_steps': MAX_TRAIN_STEPS,
                           'max_train_epochs': MAX_TRAIN_EPOCHS,
                           'img_scaling': IMG_SCALING,
                           'edge_crop': EDGE_CROP,
                           'net_scaling': NET_SCALING}
            experiment.log_multiple_params(hyperparameters)
            pipeline_dict = ml_pipeline(train_df, valid_df, hyperparameters, n_samples, input_shape)
            history = pipeline_dict["history"]
            print("Saving the best model so far")
            pipeline_dict["model"].save(BEST_MODEL_PATH)
    with experiment.test():
        # Load best model, make predictions on test data, and save them (in preparation for submission).
        model = prepare_submission(debug, BEST_MODEL_PATH, output_path)
        # TODO: Finish this 
        # loss, _ = model.evaluate(x_valid, y_valid)




if __name__ == "__main__":
    main()
