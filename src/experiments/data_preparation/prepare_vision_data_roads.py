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
    experiment_name = 'Kenya_Roads'
    train_df = pd.read_csv(f"{base_dir}/data/{experiment_name}/train_df.csv")
    validation_df = pd.read_csv(f"{base_dir}/data/{experiment_name}/validation_df.csv")
    test_df = pd.read_csv(f"{base_dir}/data/{experiment_name}/test_df.csv")

    # original kenya roads dataset
    kenya_df = pd.read_csv(f"{base_dir}/data/Kenya_Roads/roads_Kenya_256_final_dataset.csv")
    all_kenya_images_path = f"{base_dir}/data/Kenya_Roads/images"
    kenya_df = kenya_df[['ImageId', 'IRI']]
    kenya_df['class'] = kenya_df.apply(lambda row: roadsBinaryClassLabel(row['IRI']), axis=1)

    # rename ImageId to filename
    kenya_df.rename(columns={'ImageId': 'filename'}, inplace=True)
    kenya_df['filename_path'] = kenya_df['filename'].apply(lambda x: os.path.join(all_kenya_images_path, x))

    # sample 80% of the labels from the class column for training. should be stratified
    no_of_bad_roods = int(kenya_df[kenya_df['class'] == 'bad'].shape[0] * 0.8) # bad roads are the smallest
    '''
     I want the bigger class (in the case, the good class ) should be 1.3 bigger that the smaller class
    '''
    no_of_good_roads = int(no_of_bad_roods * 1.5)
    good_roads = kenya_df[kenya_df['class'] == 'good'].sample(no_of_good_roads)
    bad_roads = kenya_df[kenya_df['class'] == 'bad'].sample(no_of_bad_roods)
    train_split_df = pd.concat([good_roads, bad_roads], ignore_index=True).reset_index(drop=True)
    # sample 20% of the labels from the class column for validation. should be stratified. The classes are 'good' and 'bad'
    train_split_df['is_valid'] = np.random.choice([0, 1], size=(len(train_split_df),), p=[0.8, 0.2])
    train_split_df['is_valid'] = train_split_df['is_valid'].apply(lambda x: bool(x))
    validation_df = train_split_df[train_split_df['is_valid'] == True]
    train_df = train_split_df[train_split_df['is_valid'] == False]

    # drop the is_valid column
    validation_df.drop(columns=['is_valid'], inplace=True)
    train_df.drop(columns=['is_valid'], inplace=True)

    # shuffle the data
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    validation_df = validation_df.sample(frac=1).reset_index(drop=True)
    test_df = kenya_df[~kenya_df['filename'].isin(train_split_df['filename'])]

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

    # find the unique classes
    classes = list(train_df['class'].unique())

    # create directories inside each directory corresponding to the classes
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