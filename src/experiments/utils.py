import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import os
from tensorflow.keras.applications import ResNet50, MobileNetV2, Xception, VGG16
# import cv2
# import tensorflow_model_optimization as tfmot
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
from tensorflow.keras import layers
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomResizedCrop, Resize, ToTensor
import torchvision.transforms as transforms
import torch
import evaluate
from scipy.special import softmax
from sklearn import metrics
import datasets
import time
from transformers import ViTFeatureExtractor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
from transformers import ViTForImageClassification, DeiTForImageClassification, ConvNextForImageClassification
from transformers import DefaultDataCollator, TrainingArguments, Trainer, TrainerCallback
import evaluate
from datasets import disable_caching
from copy import deepcopy


# Balanced Weights: Treats all objectives equally.
# Accuracy-Focused: Prioritizes accuracy (MCC) over environmental impacts.
# Environment-Focused: Emphasizes minimizing carbon emissions and energy consumption.
# Energy Efficiency Priority: Focuses on reducing energy consumption.
# Carbon Emissions Priority: Focuses on minimizing carbon emissions.

"""
weight_combinations = {
    'Balanced_Weights': {'mcc': 0.33, 'carbon_emissions': 0.33, 'energy_consumed': 0.34},
    'Accuracy_Focused': {'mcc': 0.5, 'carbon_emissions': 0.25, 'energy_consumed': 0.25},
    'Environment_Focused': {'mcc': 0.2, 'carbon_emissions': 0.4, 'energy_consumed': 0.4},
    'Energy_Efficiency_Priority': {'mcc': 0.3, 'carbon_emissions': 0.3, 'energy_consumed': 0.4},
    'Carbon_Emissions_Priority': {'mcc': 0.3, 'carbon_emissions': 0.5, 'energy_consumed': 0.2},
}
"""
weight_combinations = {
    'Balanced_Weights': {'mcc': 0.34, 'carbon_emissions': 0.33, 'energy_consumed': 0.33},
    'Accuracy_Focused': {'mcc': 0.70, 'carbon_emissions': 0.15, 'energy_consumed': 0.15},
    'Environment_Focused': {'mcc': 0.2, 'carbon_emissions': 0.4, 'energy_consumed': 0.4},
    'Energy_Efficiency_Priority': {'mcc': 0.3, 'carbon_emissions': 0.2, 'energy_consumed': 0.5},
    'Carbon_Emissions_Priority': {'mcc': 0.3, 'carbon_emissions': 0.5, 'energy_consumed': 0.2},
}

# batch_size = 16
learning_rate = 1e-6
num_epochs = 100
IMAGE_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE
base_dir = "/src"
cadcarbon_dir = f"{base_dir}/results/carbon_tracker/"
results_dir = f"{base_dir}/results/metrics/"
priorities = [*weight_combinations.keys()]
# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

metric = evaluate.load("accuracy", load_from_cache_file=False)
auroc_metric = evaluate.load("roc_auc", load_from_cache_file=False)
disable_caching()

def save_prediction_metrics(metrics_df, model_name, unique_id, experiment_name, results_dir, quant, _type):
    metricsname2save = f"{results_dir}/{quant}_{experiment_name}_{model_name}_{unique_id}.csv"
    metrics_df.to_csv(metricsname2save, index=False)


def save_trained_model(model, model_name, unique_id, experiment_name, base_dir, _type):
    modelname2save = f"{base_dir}/models/cnn_models/model_{_type}_{experiment_name}_{model_name}_{unique_id}.h5"
    model.save(modelname2save)


def valid_filenames(df):
    valid_mask_df = (df['filename_path'].apply(os.path.exists) & df['filename_path'].str.endswith('.tif') & df[
        'filename_path'].apply(os.path.getsize) > 0)
    return df[valid_mask_df]


def calc_layers_to_freeze(_model_name):
    """
    Calculate the number of layers to freeze
    :param _model_name:
    :return:
    """
    layers2freeze = 0
    if _model_name == 'ResNet50':
        layers2freeze = 9
    elif _model_name == 'VGG16':
        layers2freeze = 3
    elif _model_name == 'MobileNetV2':
        layers2freeze = 10
    elif _model_name == 'Xception':
        layers2freeze = 100
    return layers2freeze


def compile_model(model_to_compile, learning_rate=None, num_of_classes=2):
    """
    Re-usable function to Compile the model
    :param model_to_compile: name of the model
    :param learning_rate: learning rate. Default is 0.001
    :param num_of_classes: number of classes. Default is 2
    :return: complied model
    """
    model_to_compile.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy' if num_of_classes == 2 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    return model_to_compile


def create_base_model(model_name, image_size, model_weights=None):
    """
    Download pre-trained models from keras applications and add a custom output layer
    :param model_name: name of the model to be downloaded
    :param image_size: size of the images
    :param num_of_classes: 2 for binary classification and > 2 for multi-class classification
    :param weights: either imagenet or None(for random initialization)
    :return: model
    """
    if model_name == 'ResNet50':
        base_model = ResNet50(weights=model_weights, include_top=False, input_shape=image_size + (3,))
    elif model_name == 'VGG16':
        base_model = VGG16(weights=model_weights, include_top=False, input_shape=image_size + (3,))
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights=model_weights, include_top=False, input_shape=image_size + (3,))
    elif model_name == 'Xception':
        base_model = Xception(weights=model_weights, include_top=False, input_shape=image_size + (3,))
    else:
        raise ValueError("Invalid model name")
    return base_model


def create_final_model(base_model, num_classes, num_to_freeze, freezeLastLayers):
    """
    :param base_model:
    :param num_classes:
    :param layers_to_freeze: number of layers to freeze
    :param freezeLastLayers: True if you want to freeze the last layers else False
    :return:
    """
    # Freeze all layers by default
    for layer in base_model.layers:
        layer.trainable = False
    # if you want to freeze some layers
    if freezeLastLayers:
        for layer in base_model.layers[-num_to_freeze:]:
            layer.trainable = True
    else:
        for layer in base_model.layers:
            layer.trainable = True

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions = tf.keras.layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation='sigmoid' if num_classes == 2 else 'softmax'
    )(x)
    _model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    # final_model = compile_model(final_model, learning_rate=0.001, num_of_classes=num_classes)
    return _model


def print_layer_trainability(model):
    """
    Helper function to print number of  trainable layers
    :param model:
    :return:
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):  # Check for convolutional layers
            if not layer.trainable:
                print(f"Layer name: {layer.name} - Frozen")
            else:
                print(f"Layer name: {layer.name} - Trainable")


def roc_auc_score_multiclass(actual_class_labels, predicted_class_labels):
    unique_class = set(actual_class_labels)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        new_actual_class = [0 if x in other_class else 1 for x in actual_class_labels]
        new_pred_class = [0 if x in other_class else 1 for x in predicted_class_labels]
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average='weighted')
        roc_auc_dict[per_class] = roc_auc
    mean_roc = np.mean([*roc_auc_dict.values()])
    return mean_roc


""""
def compute_prediction_metrics(model_name, predicted_proba, test_dataset, num_of_classes):
    # predicted_labels = np.round(predicted_proba).ravel()
    predicted_labels = np.argmax(predicted_proba, axis=1)
    actual_labels = test_dataset.labels

    accuracy = accuracy_score(actual_labels, predicted_labels)
    if num_of_classes == 2:
        auroc = roc_auc_score(actual_labels, predicted_proba)
    else:
        auroc = roc_auc_score_multiclass(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels)
    recall = recall_score(actual_labels, predicted_labels)

    # add these metrics to a dataframe
    metrics_df = pd.DataFrame({
        "model": model_name,
        "accuracy": accuracy,
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }, index=[0])

    print("Accuracy:", accuracy)
    print("AUROC:", auroc)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    return metrics_df
"""


def compute_prediction_metrics(model_name, predicted_proba, test_dataset, num_of_classes):
    """
    Compute the metrics for the model
    :param model_name:
    :param predicted_proba:
    :param test_dataset:
    :param num_of_classes:
    :return:
    """
    # predicted_labels = np.round(predicted_proba).ravel()
    predicted_labels = np.argmax(predicted_proba, axis=1)
    actual_labels = test_dataset.classes

    accuracy = accuracy_score(actual_labels, predicted_labels)
    if num_of_classes == 2:
        auroc = roc_auc_score(actual_labels, predicted_proba)
    else:
        auroc = roc_auc_score_multiclass(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels, average='weighted')
    precision = precision_score(actual_labels, predicted_labels, average='weighted')
    recall = recall_score(actual_labels, predicted_labels, average='weighted')
    metrics_df = pd.DataFrame({
        "model": model_name,
        "accuracy": accuracy,
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }, index=[0])
    print("Accuracy:", accuracy)
    print("AUROC:", auroc)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    return metrics_df


def roadsBinaryClassLabel(iriValue: np.float64, allowNeg=False):
    threshold = 7
    if 0 <= int(iriValue) <= threshold:
        labelName = 1  # 'good'
    elif int(iriValue) > threshold:
        labelName = 0  # 'bad'
    else:
        if allowNeg:
            labelName = 0
        else:
            labelName = 'invalid'
    return labelName


# --- for vision transformers ---
def custom_data_transforms(feature_extractor):
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_data_transforms = Compose([
        Resize(size=[*feature_extractor.size.values()]),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        # Resize(size=(224, 224)),
        ToTensor(),
        normalize,
    ])

    test_data_transforms = Compose([
        Resize(size=[*feature_extractor.size.values()]),
        # Resize(size=(224, 224)),

        ToTensor(),
        normalize,
    ])

    def preprocess_train_ds(current_batch):
        current_batch["pixel_values"] = [train_data_transforms(image.convert("RGB")) for image in
                                         current_batch["image"]]
        return current_batch

    def preprocess_test_ds(current_batch):
        current_batch["pixel_values"] = [test_data_transforms(image.convert("RGB")) for image in current_batch["image"]]
        return current_batch

    return preprocess_train_ds, preprocess_test_ds


class CustomCallback(TrainerCallback):
    """
    A custom callback that logs all evaluation metrics at each epoch.
    credit: https://stackoverflow.com/questions/67457480/how-to-get-the-accuracy-per-epoch-or-step-for-the-huggingface-transformers-train
    """

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def compute_metrics(prediction_metrics):
    predictions = np.argmax(prediction_metrics.predictions, axis=1)
    return metric.compute(predictions=predictions, references=prediction_metrics.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
