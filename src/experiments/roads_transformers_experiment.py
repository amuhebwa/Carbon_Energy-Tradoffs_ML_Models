import code
import pandas as pd
import tensorflow as tf
import torch.nn.functional as F
import numpy as np
from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification
from utils import *
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, TrainerCallback
import torch
import glob
import evaluate
from codecarbon import EmissionsTracker

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

if __name__ == "__main__":
    base_dir = "/Users/amuhebwa/Documents/Stanford/Research_Code/CarbonMeasure_Responsibility/src"
    carbon_metrics_dir = f"{base_dir}/results/carbon_tracker"
    results_dir = f"{base_dir}/results/metrics/"
    # common variables
    batch_size =2
    learning_rate = 2e-5
    num_epochs = 10
    """
    1. microsoft/swinv2-tiny-patch4-window16-256 | microsoft/swinv2-small-patch4-window8-256
    2. facebook/deit-tiny-patch16-224 | facebook/deit-small-patch16-224
    3. facebook/convnext-tiny-224 | facebook/convnext-small-224
    4. WinKawaks/vit-tiny-patch16-224 | WinKawaks/vit-small-patch16-224
    """
    model_checkpoint = "WinKawaks/vit-small-patch16-224"
    model_name = model_checkpoint.split("/")[-1].replace('-','_')
    experiment_name = 'Kenya_Roads'

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)


    train_images_path = f"{base_dir}/vision_data/{experiment_name}/train"
    validation_images_path = f"{base_dir}/vision_data/{experiment_name}/validation"
    test_images_path = f"{base_dir}/vision_data/{experiment_name}/test"

    train_ds = load_dataset("imagefolder", data_dir=train_images_path)
    validate_ds = load_dataset("imagefolder", data_dir=validation_images_path)
    test_ds = load_dataset("imagefolder", data_dir=test_images_path)

    train_dataset = train_ds['train']
    validate_dataset = validate_ds['train']
    test_dataset = test_ds['train']

    labels = train_dataset.features["label"].names
    num_of_classes = len(labels)
    label2id = {value: index for index, value in enumerate(labels)}
    id2label = {value: key for key, value in label2id.items()}
    preprocess_train_dataset, preprocess_test_dataset = custom_data_transforms(feature_extractor)
    train_dataset.set_transform(preprocess_train_dataset)
    validate_dataset.set_transform(preprocess_train_dataset)
    test_dataset.set_transform(preprocess_test_dataset)
    del preprocess_train_dataset, preprocess_test_dataset

    model = AutoModelForImageClassification.from_pretrained(model_checkpoint, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)

    # ------------------------------------------------------------------------------------------------------------------
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # for cuda
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    grad_checkpoint = True if "vit" in model_name else False

    training_args = TrainingArguments(
        output_dir=f"{base_dir}/models/transformer_models/{experiment_name}_{model_name}",
        # logging_dir=f"{results_dir}/logs/{experiment_name}_{model_name}.txt",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=grad_checkpoint,
        eval_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,  # 1e-6 (seems to be working best), #0.01,
        warmup_ratio=0.1,
        logging_steps=10,  # initially 10,
        eval_steps=1,  # initially not there
        logging_strategy="epoch",  # initially not there
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

    carbon_name2save = f"{carbon_metrics_dir}/{experiment_name}_{model_name}_carbon_measurements.csv"
    carbon_measurement = EmissionsTracker(output_file=carbon_name2save, log_level="error")
    carbon_measurement.start()
    train_results = trainer.train()
    model_emissions = carbon_measurement.stop()

    # evaluate the model
    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    # save the trained model
    # save_trained_model(model, model_name, experiment_name, base_dir, _type='transformer')
    model_save_name = f"{base_dir}/trained_models/transformer_models/{experiment_name}_{model_name}_model"
    trainer.save_model(model_save_name)
    model.eval()

    predicted_labels, actual_labels, predicted_probabilities = [], [], []
    for image_data in test_dataset:
        image = image_data['image']
        actual_label = image_data['label']
        pixel_values = image_data['pixel_values']
        image = pixel_values.unsqueeze(0)
        with torch.no_grad():
            image = image.to(device)
            outputs = model(pixel_values=image)
            logits = outputs.logits
            predict_probabilities = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
            predicted_label = np.argmax(predict_probabilities, axis=0)
            label_probability = predict_probabilities[predicted_label]
            predicted_labels.append(predicted_label)
            actual_labels.append(actual_label)
            predicted_probabilities.append(label_probability)
            # print(f"actual_label: {actual_label}, predicted_label: {predicted_label}, label_probability: {label_probability}")
    # compute the metrics
    accuracy_metric = evaluate.load("accuracy", load_from_cache_file=False)
    auroc_metric = evaluate.load("roc_auc", load_from_cache_file=False)
    mcc_metric = evaluate.load("matthews_correlation", load_from_cache_file=False)
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    mcc = mcc_metric.compute(predictions=predicted_labels, references=actual_labels)
    accuracy = metric.compute(predictions=predicted_labels, references=actual_labels)
    if num_of_classes == 2:
        auroc = auroc_metric.compute(prediction_scores=predicted_probabilities, references=actual_labels)
    else:
        print("AUROC is not applicable for multi-class classification : Fix this!!!!")
    f1 = f1_metric.compute(predictions=predicted_labels, references=actual_labels)
    precision = precision_metric.compute(predictions=predicted_labels, references=actual_labels)
    recall = recall_metric.compute(predictions=predicted_labels, references=actual_labels)

    accuracy = accuracy['accuracy']
    auroc = auroc['roc_auc']
    mcc = mcc['matthews_correlation']
    f1 = f1['f1']
    precision = precision['precision']
    recall = recall['recall']

    # create a dataframe to store the metrics
    metrics_df = pd.DataFrame({"experiment_name": [experiment_name], "model_name": [model_name], "accuracy": [accuracy],
                                "auroc": [auroc], "mcc": [mcc], "f1": [f1], "precision": [precision], "recall": [recall],
                                "carbon_emissions": [model_emissions]})
    metrics_df.to_csv(f"{results_dir}/{experiment_name}_{model_name}_metrics.csv", index=False)


    print(f"Accuracy: {accuracy}, AUROC: {auroc}, MCC: {mcc}, F1: {f1}, Precision: {precision}, Recall: {recall}, Carbon Emissions: {model_emissions}")


    # free up memory
    del model, trainer, train_dataset, validate_dataset, test_dataset,feature_extractor, labels, label2id, id2label, model_checkpoint, model_name, training_args, train_results

    # code.interact(local=locals())
