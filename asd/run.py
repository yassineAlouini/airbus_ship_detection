from keras.optimizers import Adam

from asd.callbacks import CALLBACKS
from asd.conf import (EDGE_CROP, IMG_SCALING, IMG_SIZE, MAX_TRAIN_EPOCHS,
                      MAX_TRAIN_STEPS, NET_SCALING, VALID_IMG_COUNT)
from asd.losses_metrics import METRICS, custom_dice_loss
from asd.models.u_net import build_u_net_model
from asd.preprocessing import create_aug_gen, get_data, make_image_gen

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
    valid_img_count = hyperparameters["valid_img_count"]
    augment_brightness = hyperparameters["augment_brightness"]
    steps_per_epoch = min(max_train_steps, n_samples // batch_size)
    print("Using {} steps per epoch.".format(steps_per_epoch))
    img_genarator = make_image_gen(train_df, batch_size, img_scaling)
    augmented_img_generator = create_aug_gen(img_genarator, augment_brightness)
    # TODO: Improve the names of these returned values.
    valid_x, valid_y = next(make_image_gen(valid_df, valid_img_count, img_scaling))
    model = get_compiled_model(hyperparameters, input_shape)
    # Use only one worker for thread-safety reason.
    # TODO: Investigate this claim.
    history = model.fit_generator(augmented_img_generator, steps_per_epoch=steps_per_epoch,
                                  epochs=max_train_epochs, validation_data=(valid_x, valid_y),
                                  callbacks=CALLBACKS, workers=1)
    return {"history": history.history, "model": model}


if __name__ == "__main__":
    train_df, valid_df = get_data()
    n_samples = train_df.shape[0]
    input_shape = IMG_SIZE
    # Default hyperparameters.
    # TODO: Use hyperopt once the whole pipeline works as expected.
    hyperparameters = {'gaussian_noise': 0.1,
                       'batch_size':  8,
                       'upsample_mode': "DECONV",
                       'augment_brightness': True,
                       'max_train_steps': MAX_TRAIN_STEPS,
                       'max_train_epochs': MAX_TRAIN_EPOCHS,
                       'valid_img_count': VALID_IMG_COUNT,
                       'img_scaling': IMG_SCALING,
                       'edge_crop': EDGE_CROP,
                       'net_scaling': NET_SCALING}
    pipeline_dict = ml_pipeline(train_df, valid_df, hyperparameters, n_samples, input_shape)
    print(pipeline_dict["history"])
