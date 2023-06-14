#!bin/bash
# This script is used to train multiple models with different backbones

# Usage:
# bash multi_model_train.sh <model_config_path> <splited_dataset_path>

# Take the first argument as model config path
model_config_path=$1

# Take the second argument as dataset path
dataset_path=$2

# List model configs in the folder
model_config_list=$(ls $model_config_path)

# Print out the model config list
echo "Model config list: $model_config_list"

# Train models one by one
for model_config in $model_config_list
do
    echo "Start training model: $model_config"
    python3 train.py --model_config $model_config_path/$model_config --dataset_path $dataset_path --with_wandb
done