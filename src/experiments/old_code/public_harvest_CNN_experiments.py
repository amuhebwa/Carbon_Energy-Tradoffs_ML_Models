import code

import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    parser = argparse.ArgumentParser(description='File and Model Parameters')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_weights', required=True)
    parser.add_argument('--freezeLastLayers', required=True)
    parser.add_argument('--experiment_name', required=True)

    args = parser.parse_args()
    model_name: str = args.model_name # ResNet50, MobileNetV2, Xception, VGG16
    model_weights: str = args.model_weights # None or 'imagenet'
    freezeLastLayers: bool = args.freezeLastLayers
    experiment_name: str = args.experiment_name # e.g., PublicHarvestNet, EuroSAT, DeepWeeds, Roads

    model_name = str(model_name)
    model_weights = str(model_weights)
    freezeLastLayers = bool(freezeLastLayers)
    experiment_name = str(experiment_name)


    image_size = (224, 224)
    num_of_classes = 2
    unique_id = 'imagenetweights' if freezeLastLayers else 'randomweights'
    train_df = pd.read_csv(f"{base_dir}/data/PublicHarvestNet/train_df.csv")
    validation_df = pd.read_csv(f"{base_dir}/data/PublicHarvestNet/validation_df.csv")
    test_df = pd.read_csv(f"{base_dir}/data/PublicHarvestNet/test_df.csv")

    train_images = f"{base_dir}/data/PublicHarvestNet/train"
    validation_images = f"{base_dir}/data/PublicHarvestNet/validation"
    test_images = f"{base_dir}/data/PublicHarvestNet/test"

    # append the directory to the filename
    train_df['filename_path'] = train_df['filename'].apply(lambda x: os.path.join(train_images, x))
    validation_df['filename_path'] = validation_df['filename'].apply(lambda x: os.path.join(validation_images, x))
    test_df['filename_path'] = test_df['filename'].apply(lambda x: os.path.join(test_images, x))

    train_df = valid_filenames(train_df)
    validation_df = valid_filenames(validation_df)
    test_df = valid_filenames(test_df)

    """
    The "flow_from_dataframe" method expects the y_colum to be named 'class' and must be a string.
    """
    train_df['class'] = train_df['class'].apply(lambda x: str(x))
    validation_df['class'] = validation_df['class'].apply(lambda x: str(x))
    test_df['class'] = test_df['class'].apply(lambda x: str(x))

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, horizontal_flip=True, )
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_dataset = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_images,
        x_col="filename",
        y_col="class",
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary' if num_of_classes == 2 else 'categorical',
    )
    validation_dataset = validation_datagen.flow_from_dataframe(
        dataframe=validation_df,
        directory=validation_images,
        x_col="filename",
        y_col="class",
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary' if num_of_classes == 2 else 'categorical',
    )
    test_dataset = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=test_images,
        x_col="filename",
        y_col="class",
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary' if num_of_classes == 2 else 'categorical',
    )

    """
    These are the sets of experiments that we are going to run
    models = [ResNet50, MobileNetV2, Xception, VGG16]
    1. Retrain the model from scratch
    2. Fine-tune the model by unfreezing the last k layers and imagenet weights
    3. Fine-tune the model by unfreezing the last k layers and using eurosat weights or weights from a remote sensing dataset
    
    """
    # create the base model
    num_layers_to_freeze = calc_layers_to_freeze(model_name)
    base_model = create_base_model(model_name, image_size, model_weights)

    model = create_final_model(base_model, num_of_classes, num_layers_to_freeze, freezeLastLayers)
    model = compile_model(model, learning_rate, num_of_classes=num_of_classes)

    carbonname2save = f"{cadcarbon_dir}/unquantized_{experiment_name}_{model_name}_{unique_id}.csv"
    first_model_carbon = EmissionsTracker(output_file=carbonname2save,log_level="error")
    first_model_carbon.start()
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=validation_dataset, verbose=1,
                        callbacks=[early_stopping])
    first_model_emmissions = first_model_carbon.stop()
    predicted_proba = model.predict(test_dataset)

    metrics_df = compute_prediction_metrics(model_name, predicted_proba, test_dataset, num_of_classes)
    metrics_df['model_name'] = model_name
    metrics_df['unique_id'] = unique_id
    metrics_df['model_type'] = 'unquantized'
    metrics_df['carbon_emissions'] = first_model_emmissions

    # save the trained model
    save_trained_model(model, model_name, unique_id, experiment_name, base_dir, _type='CNN')
    # save metrics for the unquantized model
    save_prediction_metrics(metrics_df, model_name, unique_id, experiment_name, results_dir, 'unquantized', 'CNN')

    # Quantize the model
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = create_final_model(base_model, num_of_classes, num_layers_to_freeze, freezeLastLayers)
    q_aware_model = quantize_model(q_aware_model)
    q_aware_model = compile_model(q_aware_model, learning_rate, num_of_classes=num_of_classes)
    # train the quantized model
    carbonname2save = f"{cadcarbon_dir}/quantized_{experiment_name}_{model_name}_{unique_id}.csv"
    second_model_carbon = EmissionsTracker(output_file=carbonname2save, log_level="error")
    second_model_carbon.start()
    quant_history = q_aware_model.fit(train_dataset, epochs=num_epochs, validation_data=validation_dataset, verbose=1,
                                      callbacks=[early_stopping])
    second_model_emmissions = second_model_carbon.stop()

    quant_predicted_proba = q_aware_model.predict(test_dataset)
    quant_metrics_df = compute_prediction_metrics(model_name, predicted_proba, test_dataset, num_of_classes)
    quant_metrics_df['model_name'] = model_name
    quant_metrics_df['unique_id'] = unique_id
    quant_metrics_df['model_type'] = 'quantized'
    quant_metrics_df['carbon_emissions'] = second_model_emmissions

    combined_metrics_df = pd.concat([metrics_df, quant_metrics_df], axis=0)
    # save metrics for the quantized models
    save_prediction_metrics(combined_metrics_df, model_name, unique_id, experiment_name, results_dir, 'quantized',
                            'CNN')