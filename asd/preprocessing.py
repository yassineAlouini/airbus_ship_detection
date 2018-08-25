# Some libraries imports


import os

import numpy as np
import pandas as pd
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

    print("Processed ships DataFrame")
    print(ships_df.head())

    # Split the ships DataFrame into train and validation DataFrames. Notice the use of the
    # stratify keyword: this is needed so that train and validation datasets have similar number of ships
    # distributions. Otherwise, train and validation phases won't be comparable.
    train_ids, valid_ids = train_test_split(ships_df, test_size=valid_size, stratify=ships_df['ships'])
    train_df = pd.merge(masks_df, train_ids)
    valid_df = pd.merge(masks_df, valid_ids)

    print(train_df.shape[0], "Training masks (befre rebalancing) DataFrame rows")
    print(valid_df.shape[0], "Validation masks DataFrame rows")

    # In this final step, we reblanace the training DataFrame by undersampling from the "no ships" class.
    # Notice that the +2 is done to have the "no ships" class in a category by itself.
    # The // division creates buckets for grouping images by the number of ships present: images with 4 and 5 ships
    # will be put under the same bucket. Finally, the rebalanced dataset could be smaller than the original one if
    # if number of unique ships buckets * SAMPLES_PER_SHIPS_BUCKET < number of rows. It could be bigger as well.
    # In that case, you need to set replace in the sample function to True.
    # TODO: Find a way to replace the .apply with a vectorized operation.
    balanced_train_df = (train_df.assign(ships_bucket=lambda df: (df["ships"] + 2) // 3)
                                 .groupby('ships_bucket')
                                 .apply(lambda x: x.sample(SAMPLES_PER_SHIPS_BUCKET,
                                                           random_state=SEED)))
    return balanced_train_df, valid_df


if __name__ == "__main__":
    train_df, valid_df = get_data()
    print(train_df.head())
    print(train_df.shape)
    print(valid_df.head())
    print(valid_df.shape)
