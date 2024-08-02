import code

import code
import shutil
import numpy as np
import tensorflow as tf
import os
import pandas as pd

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

def valid_filenames(df):
    valid_mask_df = (df['filename_path'].apply(os.path.exists) & (df['filename_path'].str.endswith('.tif') | df['filename_path'].str.endswith('.jpg'))& df['filename_path'].apply(os.path.getsize) > 0)
    return df[valid_mask_df]


def roadsBinaryClassLabel(iriValue: np.float64, allowNeg=False):
    threshold = 7
    if 0 <= int(iriValue) <= threshold:
        labelName = 'good'  # 'good'
    elif int(iriValue) > threshold:
        labelName = 'bad'  # 'bad'
    else:
        if allowNeg:
            labelName = 0
        else:
            labelName = 'invalid'
    return labelName


if __name__ == "__main__":

    base_dir = "/Users/amuhebwa/Documents/Stanford/Research_Code/CarbonMeasure_Responsibility/src"
    experiment_name = 'EuroSAT'
    train_df = pd.read_csv(f"{base_dir}/data/{experiment_name}/train_df.csv")
    validation_df = pd.read_csv(f"{base_dir}/data/{experiment_name}/validation_df.csv")
    test_df = pd.read_csv(f"{base_dir}/data/{experiment_name}/test_df.csv")

    # directory where the train, validation and test images are stored
    train_images_path = f"{base_dir}/data/{experiment_name}/train"
    validation_images_path = f"{base_dir}/data/{experiment_name}/validation"
    test_images_path = f"{base_dir}/data/{experiment_name}/test"

    # append the directory to the filename
    train_df['filename_path'] = train_df['filename'].apply(lambda x: os.path.join(train_images_path, x))
    validation_df['filename_path'] = validation_df['filename'].apply(lambda x: os.path.join(validation_images_path, x))
    test_df['filename_path'] = test_df['filename'].apply(lambda x: os.path.join(test_images_path, x))

    # concatenate the train, validation and test dataframes
    combined_df = pd.concat([train_df, validation_df, test_df], ignore_index=True).reset_index(drop=True)
    combined_df = valid_filenames(combined_df)
    combined_df = combined_df.sample(frac=1.0).reset_index(drop=True)

    classes = list(combined_df['class'].unique())
    train_dfs, test_dfs, validation_dfs = [], [], []
    for cls in classes:
        temp_df = combined_df[combined_df['class'] == cls]
        total_labels = temp_df.shape[0]
        train_size = int(total_labels * 0.6)

        train_split_df = temp_df.sample(train_size, replace=False)
        current_train_df = train_split_df.sample(frac=0.7, random_state=seed, replace=False)
        current_validation_df = train_split_df[~train_split_df['filename'].isin(current_train_df['filename'])]
        # use filename to get the test_df
        current_test_df = temp_df[~temp_df['filename'].isin(train_split_df['filename'])]

        # we don't have enough computation resources so we are going to sample the training and validation sets
        # generate a random integer between 200 and 400.
        # k = np.random.randint(200, 400)
        #train_sampled_count = min(np.random.randint(200, 400), len(current_train_df))
        #validation_sampled_count = min(np.random.randint(200, 400), len(current_validation_df))
        #current_train_df = current_train_df.sample(train_sampled_count, replace=False)
        #current_validation_df = current_validation_df.sample(validation_sampled_count, replace=False)
        train_dfs.append(current_train_df)
        test_dfs.append(current_test_df)
        validation_dfs.append(current_validation_df)

    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1.0).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1.0).reset_index(drop=True)
    validation_df = pd.concat(validation_dfs, ignore_index=True).sample(frac=1.0).reset_index(drop=True)

    train_df = valid_filenames(train_df)
    validation_df = valid_filenames(validation_df)
    test_df = valid_filenames(test_df)

    train_images_destination = f"{base_dir}/vision_data/{experiment_name}/train"
    validation_images_destination = f"{base_dir}/vision_data/{experiment_name}/validation"
    test_images_destination = f"{base_dir}/vision_data/{experiment_name}/test"

    # create for the train, validation and test directories
    os.makedirs(train_images_destination, exist_ok=True)
    os.makedirs(validation_images_destination, exist_ok=True)
    os.makedirs(test_images_destination, exist_ok=True)

    for c in classes:
        os.makedirs(f"{train_images_destination}/{c}", exist_ok=True)
        os.makedirs(f"{validation_images_destination}/{c}", exist_ok=True)
        os.makedirs(f"{test_images_destination}/{c}", exist_ok=True)

    # loop through the dataframes and copy the images to the respective directories
    for index, row in train_df.iterrows():
        src_name = row['filename_path']
        dest_name = f"{train_images_destination}/{row['class']}"
        shutil.copy(src_name, dest_name)

    for index, row in validation_df.iterrows():
        src_name = row['filename_path']
        dest_name = f"{validation_images_destination}/{row['class']}"
        shutil.copy(src_name, dest_name)

    for index, row in test_df.iterrows():
        src_name = row['filename_path']
        dest_name = f"{test_images_destination}/{row['class']}"
        shutil.copy(src_name, dest_name)

    print("Done copying files to train, validation and test directories")
