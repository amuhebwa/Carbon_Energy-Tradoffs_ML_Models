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
from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification
from src.experiments.utils import *
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, TrainerCallback
import torch

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

if __name__ == "__main__":
    base_dir = "/src"
    images_dir = f"{base_dir}/data/EuroSAT/images/"
    cadcarbon_dir = f"{base_dir}/results/carbon_tracker/"
    results_dir = f"{base_dir}/results/metrics/"

    # common variables
    model_weights = 'imagenet'  # None or 'imagenet'
    freezeLastLayers = True  # true if you are fine-tuning the model
    image_size = (224, 224)
    batch_size =4
    num_of_classes = 10
    learning_rate = 1e-3
    num_epochs = 5
    experiment_name = 'EuroSAT'
    unique_id = 'imagenetweights' if freezeLastLayers else 'randomweights'
    model_checkpoint = "google/vit-large-patch16-224"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    # train_df = pd.read_csv(f"{base_dir}/data/EuroSAT/train_df.csv")
    #validation_df = pd.read_csv(f"{base_dir}/data/EuroSAT/validation_df.csv")
    #test_df = pd.read_csv(f"{base_dir}/data/EuroSAT/test_df.csv")

    train_images_path = f"{base_dir}/vision_data/EuroSAT/train"
    validation_images_path = f"{base_dir}/vision_data/EuroSAT/validation"
    test_images_path = f"{base_dir}/vision_data/EuroSAT/test"

    # num_of_classes = len(train_df['class'].unique())

    train_ds = load_dataset("imagefolder", data_dir=train_images_path)
    validate_ds = load_dataset("imagefolder", data_dir=validation_images_path)
    test_ds = load_dataset("imagefolder", data_dir=test_images_path)

    train_dataset = train_ds['train']
    validate_dataset = validate_ds['train']
    test_dataset = test_ds['train']


    del train_ds, validate_ds, test_ds, train_images_path, validation_images_path, test_images_path

    labels = train_dataset.features["label"].names
    label2id = {'Residential': 0, 'AnnualCrop': 1, 'Highway': 2, 'SeaLake': 3, 'HerbaceousVegetation': 4,
                'Pasture': 5, 'Industrial': 6, 'PermanentCrop': 7, 'Forest': 8, 'River': 9}
    id2label = {value: key for key, value in label2id.items()}

    preprocess_train_dataset, preprocess_test_dataset = custom_data_transforms(feature_extractor)
    train_dataset.set_transform(preprocess_train_dataset)
    validate_dataset.set_transform(preprocess_train_dataset)
    test_dataset.set_transform(preprocess_test_dataset)

    model = AutoModelForImageClassification.from_pretrained(model_checkpoint, label2id=label2id, id2label=id2label,
                                                           ignore_mismatched_sizes=True)



    # comment this out if you are using mac os
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model.to(device)

    # code.interact(local=locals())

    model_name = model_checkpoint.split("/")[-1]
    # improving performance: https://huggingface.co/docs/transformers/v4.18.0/en/performance
    training_args = TrainingArguments(
        output_dir=f"{base_dir}/models/transformer_models/{experiment_name}_{unique_id}_{model_name}",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        eval_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,  # 1e-6 (seems to be working best), #0.01,
        warmup_ratio=0.1,
        logging_steps=10,  # initially 10,
        eval_steps=1,  # initially not there
        logging_strategy="epoch",  # initially not there
        # do_train=True,  # initially not there
        # do_eval=True,  # initially not there
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validate_dataset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    # Extra call back to log training accuracy
    trainer.add_callback(CustomCallback(trainer))
    train_results = trainer.train()

