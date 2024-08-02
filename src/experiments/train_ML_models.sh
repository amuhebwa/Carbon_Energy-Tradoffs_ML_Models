#!/bin/bash

#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-long
#SBATCH --gpus=1
##SBATCH --gpus-per-task=2
#SBATCH --time=23:00:00
#SBATCH -o slurm-%j.out


source activate tf
module load cuda/11.4.2 cudnn/8.2.4.15-11.4
model_name="MobileNetV2" #ResNet50, MobileNetV2, Xception, VGG16
model_weights="imagenet"  # None or imagenet
freezeLastLayers=True
experiment_name="EuroSAT"#PublicHarvestNet, EuroSAT, DeepWeeds, Roads

srun -n1 python eurosat_CNN_experiments.py --model_name=$model_name --model_weights=$model_weights --freezeLastLayers=$freezeLastLayers --experiment_name=$experiment_name

