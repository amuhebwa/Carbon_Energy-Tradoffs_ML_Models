import code
import shutil
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from PIL import Image

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

def convert_tif_to_jpg(input_path, output_path):
    try:
        # Open the .tif image
        with Image.open(input_path) as img:
            # Convert to RGB if it's not already in that mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Save as .jpg
            img.save(output_path, 'JPEG')
        print(f"Converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
def valid_filenames(df):
    valid_mask_df = (df['filename_path'].apply(os.path.exists) & df['filename_path'].str.endswith('.tif'))
    return df[valid_mask_df]

if __name__ == "__main__":

    base_dir = "/Users/amuhebwa/Documents/Stanford/Research_Code/CarbonMeasure_Responsibility/src"
    experiment_name = 'PublicHarvestNet'
    # original kenya roads dataset
    labels_df = pd.read_csv(f"{base_dir}/data/{experiment_name}/labels_all.csv")
    all_images_path = f"{base_dir}/data/{experiment_name}/skysat_images_all"

    labels_df = labels_df[['filename', 'activity']]

    # append the directory to the filename
    labels_df['filename_path'] = labels_df['filename'].apply(lambda x: os.path.join(all_images_path, x))
    # convert activity to integer
    labels_df['activity'] = labels_df['activity'].apply(lambda x: str(int(x)))

    # rename activity to class
    labels_df = labels_df.rename(columns={'activity': 'class'})

    classes = list(labels_df['class'].unique())
    train_dfs, test_dfs, validation_dfs = [], [], []
    for cls in classes:
        temp_df = labels_df[labels_df['class'] == cls]
        total_labels = temp_df.shape[0]
        train_size = int(total_labels * 0.8)
        train_split_df = temp_df.sample(train_size, replace=False)
        current_train_df = train_split_df.sample(frac=0.7, random_state=seed, replace=False)
        current_validation_df = train_split_df[~train_split_df['filename'].isin(current_train_df['filename'])]
        # use filename to get the test_df
        current_test_df = temp_df[~temp_df['filename'].isin(train_split_df['filename'])]
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

    # create directories inside each directory corresponding to the classes
    for c in classes:
        os.makedirs(f"{train_images_destination}/class_{c}", exist_ok=True)
        os.makedirs(f"{validation_images_destination}/class_{c}", exist_ok=True)
        os.makedirs(f"{test_images_destination}/class_{c}", exist_ok=True)

    # loop through the dataframes and copy the images to the respective directories
    for index, row in train_df.iterrows():
        src_name = row['filename_path']
        dest_name = f"{train_images_destination}/class_{row['class']}/{row['filename'].replace('.tif', '.jpg')}"
        convert_tif_to_jpg(src_name, dest_name)
    for index, row in validation_df.iterrows():
        src_name = row['filename_path']
        dest_name = f"{validation_images_destination}/class_{row['class']}/{row['filename'].replace('.tif', '.jpg')}"
        convert_tif_to_jpg(src_name, dest_name)

    for index, row in test_df.iterrows():
        src_name = row['filename_path']
        dest_name = f"{test_images_destination}/class_{row['class']}/{row['filename'].replace('.tif', '.jpg')}"
        convert_tif_to_jpg(src_name, dest_name)

    print("Done copying files to train, validation and test directories")


