import code

import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50, MobileNetV2, Xception, VGG16
import cv2
import tensorflow_model_optimization as tfmot
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
from codecarbon import EmissionsTracker
from src.experiments.utils import *

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

if __name__ == "__main__":
    base_dir = "/src"
    cadcarbon_dir = f"{base_dir}/results/carbon_tracker/"
    results_dir = f"{base_dir}/results/metrics/"

    # common variables
    model_weights = 'imagenet'  # None or 'imagenet'
    freezeLastLayers = True  # true if you are fine-tuning the model
    model_name = 'MobileNetV2'  # ResNet50, MobileNetV2, Xception, VGG16
    image_size = (224, 224)
    batch_size = 32
    num_of_classes = 2
    learning_rate = 1e-3
    num_epochs = 5
    experiment_name = 'EuroSAT'
    unique_id = 'imagenetweights' if freezeLastLayers else 'randomweights'

    train_df = pd.read_csv(f"{base_dir}/data/EuroSAT/train_df.csv")
    validation_df = pd.read_csv(f"{base_dir}/data/EuroSAT/validation_df.csv")
    test_df = pd.read_csv(f"{base_dir}/data/EuroSAT/test_df.csv")


    train_images = f"{base_dir}/data/EuroSAT/train"
    validation_images = f"{base_dir}/data/EuroSAT/validation"
    test_images = f"{base_dir}/data/EuroSAT/test"

    # append the directory to the filename
    train_df['filename_path'] = train_df['filename'].apply(lambda x: os.path.join(train_images, x))
    validation_df['filename_path'] = validation_df['filename'].apply(lambda x: os.path.join(validation_images, x))
    test_df['filename_path'] = test_df['filename'].apply(lambda x: os.path.join(test_images, x))

    # train_df = valid_filenames(train_df)
    # validation_df = valid_filenames(validation_df)
    # test_df = valid_filenames(test_df)

    def sample_k_elements(group, k):
        return group.sample(min(k, len(group)))  # Sample k elements or all elements if k is greater than the group size

    k = 100
    train_df = train_df.groupby('class', group_keys=False).apply(lambda group: sample_k_elements(group, k))
    validation_df = validation_df.groupby('class', group_keys=False).apply(lambda group: sample_k_elements(group, k))
    test_df = test_df.groupby('class', group_keys=False).apply(lambda group: sample_k_elements(group, k))

    """
    The "flow_from_dataframe" method expects the y_colum to be named 'class' and must be a string.
    """
    train_df['class'] = train_df['class'].apply(lambda x: str(x))
    validation_df['class'] = validation_df['class'].apply(lambda x: str(x))
    test_df['class'] = test_df['class'].apply(lambda x: str(x))

    # get unique classes
    classes = train_df['class'].unique()
    # classes = [f'class_{c}' for c in classes]

    vision_train_dir = f"{base_dir}/vision_data/{experiment_name}/train"
    vision_validation_dir = f"{base_dir}/vision_data/{experiment_name}/validation"
    vision_test_dir = f"{base_dir}/vision_data/{experiment_name}/test"

    # create directories inside each directory corresponding to the classes
    for c in classes:
        os.makedirs(f"{vision_train_dir}/{c}", exist_ok=True)
        os.makedirs(f"{vision_validation_dir}/{c}", exist_ok=True)
        os.makedirs(f"{vision_test_dir}/{c}", exist_ok=True)

    # loop through the dataframes and copy the images to the respective directories
    for index, row in train_df.iterrows():
        src_name = row['filename_path']
        # dest_name = f"{vision_train_dir}/class_{row['class']}"
        dest_name = f"{vision_train_dir}/{row['class']}"
        shutil.copy(src_name, dest_name)

    for index, row in validation_df.iterrows():
        src_name = row['filename_path']
        dest_name = f"{vision_validation_dir}/{row['class']}"
        shutil.copy(src_name, dest_name)

    for index, row in test_df.iterrows():
        src_name = row['filename_path']
        dest_name = f"{vision_test_dir}/{row['class']}"
        shutil.copy(src_name, dest_name)

    print("Data copied to vision directories")

