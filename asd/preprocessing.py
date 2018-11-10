# Some libraries imports


import os

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from asd.conf import (FILE_SIZE_KB_THRESHOLD, MASKS_DATA_PATH,
                      SAMPLES_PER_SHIPS_BUCKET, SEED, TRAIN_IMAGES_FOLDER,
                      VALID_SIZE)

"""
A data loading and rebalancing script that uses idomatic Pandas (vectorization, no apply, and so on).
Notice that some of the processing functions are inspired from this kernel:
https://www.kaggle.com/kmader/baseline-u-net-model-part-1.
"""


# Two vectorized functions

""" Vectorizing functions makes DataFrame manipulation faster. Check this blog post for more details (the vectorization section):
https://tomaugspurger.github.io/modern-4-performance.
"""
_v_path_join = np.vectorize(os.path.join)
_v_file_size = np.vectorize(lambda fp: (os.stat(fp).st_size) / 1024)


def get_data(file_size_kb_threshold=FILE_SIZE_KB_THRESHOLD,
             valid_size=VALID_SIZE):
    """ Load and filter raw data using image sizes, split into train and validation data, and finally rebalance
    the train data using the number of ships.
    """

    # Load the masks DataFrame
    masks_df = pd.read_csv(MASKS_DATA_PATH)

    print("Raw mask DataFrame")
    print(masks_df.shape)
    print(masks_df.head())

    # Count the number of ships in each mask (using the "EncodingPixels" column), create
    # the "has_ship" column (using np.where to make the if/else operation vectorized), compute the size
    # of each image (using the _v_path_join and _v_file_size vectorized functions), and finally
    # filter out rows where the file size is less than a threshold (by default 50 KO).
    ships_df = (masks_df.groupby('ImageId')["EncodedPixels"]
                        .count()
                        .reset_index()
                        .rename(columns={"EncodedPixels": "ships"})
                        .assign(has_ship=lambda df: np.where(df['ships'] > 0, 1, 0))
                        .assign(file_path=lambda df: _v_path_join(TRAIN_IMAGES_FOLDER,
                                                                  df.ImageId.astype(str)))
                        .assign(file_size_kb=lambda df: _v_file_size(df.file_path))
                        .loc[lambda df: df.file_size_kb > file_size_kb_threshold, :])

    # Only keep train data with ships (see if it improves the validation TPR).

    ships_df = ships_df.loc[lambda df: df.has_ship == 1]

    # Split the ships DataFrame into train and validation DataFrames. Notice the use of the
    # stratify keyword: this is needed so that train and validation datasets have similar number of ships
    # distributions. Otherwise, train and validation phases won't be comparable.
    train_ids, valid_ids = train_test_split(ships_df, test_size=valid_size, stratify=ships_df['ships'])
    train_df = pd.merge(masks_df, train_ids)
    valid_df = pd.merge(masks_df, valid_ids)

    print(train_df.shape[0], "Training masks (before rebalancing) DataFrame rows")
    print(valid_df.shape[0], "Validation masks DataFrame rows")

    # In this final step, we reblanace the training DataFrame by undersampling from the "no ships" class.
    # Notice that the +2 is done to have the "no ships" class in a category by itself.
    # The // division creates buckets for grouping images by the number of ships present: images with 4 and 5 ships
    # will be put under the same bucket. Finally, the rebalanced dataset could be smaller than the original one if
    # if number of unique ships buckets * SAMPLES_PER_SHIPS_BUCKET < number of rows. It could be bigger as well.
    # In that case, you need to set replace in the sample function to True.
    # TODO: Find a way to replace the .apply with a vectorized operation.
    # Try without rebalancing.
    # balanced_train_df = (train_df.assign(ships_bucket=lambda df: (df["ships"] + 2) // 3)
    #                              .groupby('ships_bucket')
    #                              .apply(lambda x: x.sample(SAMPLES_PER_SHIPS_BUCKET,
    #                                                        random_state=SEED) if len(x) > SAMPLES_PER_SHIPS_BUCKET
    #                                     else x))
    return train_df, valid_df


# TODO: Finish cleaning the next few functions.

def rle_encode(img, min_threshold=1e-3, max_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_threshold:
        return ''  # no need to encode if it's all zeros
    if max_threshold and np.mean(img) > max_threshold:
        return ''  # ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# TODO: The hardcoded shape (768, 768) should be moved to a constant.
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    # TODO: This mask size shouldn't be hardcoded (768, 768).
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    # TODO: This mask size shouldn't be hardcoded (768, 768).
    all_masks = np.zeros((768, 768), dtype=np.float)

    def scale(x): return (len(in_mask_list) + x + 1) / (len(in_mask_list) * 2)  # scale the heatmap image to shift
    for i, mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:, :] += scale(i) * rle_decode(mask)
    return all_masks

# TODO: Add some documentation


def make_image_gen(input_df, batch_size, img_scaling):
    df = input_df.copy()
    all_batches = list(df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(TRAIN_IMAGES_FOLDER, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            c_mask = np.expand_dims(c_mask, axis=-1)
            if img_scaling is not None:
                c_img = c_img[::img_scaling[0], ::img_scaling[1]]
                c_mask = c_mask[::img_scaling[0], ::img_scaling[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []

# TODO: Add some documentation for the augmentation pipeline as well.
# TODO: Finish this and add some documentation.


def build_image_generator(augment_brightness):
    """ Build an image data generator (for images and labels).
    For more details about this class, check the documentation here:
    https://keras.io/preprocessing/image/.
    """
    # TODO: Describe what each data augementation parameter does.
    data_generator_dict = dict(featurewise_center=False,
                               samplewise_center=False,
                               rotation_range=45,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode='reflect',
                               data_format='channels_last')
    # brightness can be problematic since it seems to change the labels differently from the images
    if augment_brightness:
        data_generator_dict['brightness_range'] = [0.5, 1.5]
    image_gen = ImageDataGenerator(**data_generator_dict)

    if augment_brightness:
        data_generator_dict.pop('brightness_range')
    label_gen = ImageDataGenerator(**data_generator_dict)
    return image_gen, label_gen

# TODO: Add some documentation and improve variables names.


def create_aug_gen(in_gen, augment_brightness, seed=None):
    image_gen, label_gen = build_image_generator(augment_brightness)
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255 * in_x,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        yield next(g_x) / 255.0, next(g_y)


if __name__ == "__main__":
    train_df, valid_df = get_data()
    print(train_df.head())
    print(train_df.shape)
    print(valid_df.head())
    print(valid_df.shape)
