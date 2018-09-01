import glob
import os

import click
import numpy as np
import pandas as pd
from keras import models
from keras.optimizers import Adam
from skimage.io import imread
from tqdm import tqdm

from asd.callbacks import CALLBACKS
from asd.conf import (BEST_MODEL_PATH, EDGE_CROP, IMG_SCALING, IMG_SIZE,
                      MAX_TRAIN_EPOCHS, MAX_TRAIN_STEPS, NET_SCALING,
                      TEST_IMAGES_FOLDER, TEST_IMGS_TO_IGNORE)
from asd.losses_metrics import (METRICS, custom_dice_loss, dice_metric,
                                true_positive_rate_metric)
from asd.models.u_net import build_u_net_model
from asd.preprocessing import (create_aug_gen, get_data, make_image_gen,
                               rle_encode)

# TODO: Add some documentation


def get_compiled_model(hyperparameters, input_shape=IMG_SIZE):
    model = build_u_net_model(input_shape, **hyperparameters)
    # TODO: These should be in the hp list as well.
    learning_rate = 1e-3
    decay = 1e-6
    adam_optimizer = Adam(learning_rate, decay=decay)
    model.compile(optimizer=adam_optimizer, loss=custom_dice_loss, metrics=METRICS)
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
    augmented_img_generator = create_aug_gen(img_genarator, augment_brightness)
    # TODO: Improve the names of these returned values.
    valid_x, valid_y = next(make_image_gen(valid_df, batch_size, img_scaling))
    model = get_compiled_model(hyperparameters, input_shape)
    # Use only one worker for thread-safety reason.
    # TODO: Investigate this claim.
    history = model.fit_generator(augmented_img_generator, steps_per_epoch=steps_per_epoch,
                                  epochs=max_train_epochs, validation_data=(valid_x, valid_y),
                                  callbacks=CALLBACKS, workers=1)
    return {"history": history.history, "model": model}


def prepare_submission(model_path, output_path):
    """ Load the best trained models and predict the masks for the test images.
    """
    # Loda the model
    model = models.load_model(model_path, custom_objects={
                              'custom_dice_loss': custom_dice_loss,
                              'true_positive_rate_metric': true_positive_rate_metric,
                              'dice_metric': dice_metric})
    encoded_predictions = []
    img_ids = []
    for img_path in tqdm(glob.iglob(os.path.join(TEST_IMAGES_FOLDER, "*.jpg"))):
        img_id = img_path.split('/')[-1].replace('.jpg', '')
        if img_id not in TEST_IMGS_TO_IGNORE:
            img = imread(img_path)
            img = np.expand_dims(img, 0) / 255.0
            predictions = model.predict(img)
            # Encode
            # TODO: Use the multiple_rle_encode function?
            encoded_predictions.append(rle_encode(predictions))
            img_ids.append(img_ids)
    assert len(img_ids) == len(encoded_predictions)
    # Preparing and saving the submission file.
    predictions_df = pd.DataFrame({"ImageId": img_ids, "EncodedPixels": encoded_predictions})
    print(predictions_df.head())
    predictions_df.to_csv(output_path, index=False)


@click.command()
@click.option('--debug', type=bool,
              default=True, help='Whether to run the pipeline in debug mode or not. Defaults to True.')
@click.option('--train', type=bool,
              default=False, help='Whether to train a model or load one. Defaults to False.')
@click.option('--output_path', type=str, help='Where to store the submission file.')
def main(debug, train, output_path):
    if train:
        train_df, valid_df = get_data()
        n_samples = train_df.shape[0]
        input_shape = IMG_SIZE
        # Default hyperparameters.
        # TODO: Use hyperopt once the whole pipeline works as expected.
        hyperparameters = {'gaussian_noise': 0.1,
                           'batch_size':  32,
                           'upsample_mode': "DECONV",
                           'augment_brightness': True,
                           'max_train_steps': MAX_TRAIN_STEPS,
                           'max_train_epochs': MAX_TRAIN_EPOCHS,
                           'img_scaling': IMG_SCALING,
                           'edge_crop': EDGE_CROP,
                           'net_scaling': NET_SCALING}
        pipeline_dict = ml_pipeline(train_df, valid_df, hyperparameters, n_samples, input_shape)
        print(pipeline_dict["history"])
        print("Saving the best model so far")
        pipeline_dict["model"].save(BEST_MODEL_PATH)
    # Load best model, make predictions on test data, and save them (in preparation for submission).
    prepare_submission(BEST_MODEL_PATH, output_path)


if __name__ == "__main__":
    main()
