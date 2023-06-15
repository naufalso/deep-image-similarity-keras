# Author: Naufal Suryanto (https://github.com/naufalso)

import os
import argparse
import wandb
import json

from image_similarity_keras.triplet_dataset import TripletDataset
from image_similarity_keras.model import SiameseModel


def main():
    """Main function for training the model

    Args:
        dataset_path (str): Path to dataset
        model_config (str): Path to model config file
        augmentation_config (str): Path to augmentation config file
        metrics_config (str): Path to metrics config file
        batch_size (int): Batch size for training
        optimizer (str): Optimizer to use for training
        learning_rate (float): Learning rate for training
        loss (str): Loss function to use for training
        epochs (int): Number of epochs for training
        finetune (bool): Whether to finetune the model
        output_path (str): Path to output directory
        wandb_project (str): Wandb project name
    """

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Argument parser for image similarity project"
    )

    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to model config file",
    )
    parser.add_argument(
        "--augmentation_config",
        type=str,
        default="./configs/default_augmentation.json",
        help="Path to augmentation config file",
    )
    parser.add_argument(
        "--metrics_config",
        type=str,
        default="./configs/default_metrics.json",
        help="Path to metrics config file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="TripletSemiHardLoss",
        help="Loss function to use for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs for training"
    )
    parser.add_argument(
        "--finetune", action="store_true", help="Whether to finetune the model"
    )
    parser.add_argument(
        "--output_path", type=str, default="./models/", help="Path to output directory"
    )
    parser.add_argument("--with_wandb", action="store_true", help="Use wandb or not")

    args = parser.parse_args()

    # Load model config
    with open(args.model_config, "r") as f:
        model_config = json.load(f)

        # Convert to Namespace
        model_config_ns = argparse.Namespace(**model_config)

    # Load augmentation config
    with open(args.augmentation_config, "r") as f:
        augmentation_config = json.load(f)

    # Load metrics config
    with open(args.metrics_config, "r") as f:
        metrics_config = json.load(f)

    # Initialize wandb
    if args.with_wandb:
        wandb.init(
            name=f"{model_config_ns.backbone_name}_{args.batch_size}b_{args.epochs}ep",
            config={
                "dataset": args.dataset_path,
                "model/backbone": model_config_ns.backbone_name,
                "model/image_size": model_config_ns.image_size,
                "model/activation": "relu",
                "model/fc_depths": model_config_ns.fc_depths,
                "model/emb_dims": model_config_ns.emb_dims,
                "train_config/finetune": args.finetune,
                "train_config/learning_rate": args.learning_rate,
                "train_config/optimizer": args.optimizer,
                "train_config/loss": args.loss,
                "train_config/batch_size": args.batch_size,
                "train_config/epochs": args.epochs,
            },
        )

    # Initialize dataset generator
    triplet_dataset = TripletDataset(
        args.dataset_path, args.batch_size, model_config_ns.image_size
    )

    dataset_generators, dataset_steps_per_epoch = triplet_dataset.get_triplet_generator(
        **augmentation_config
    )

    # Initialize model
    model = SiameseModel(**model_config)

    # Build and compile model
    model.build(args.finetune).compile(
        args.batch_size, args.optimizer, args.learning_rate, args.loss, metrics_config
    )

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Train model
    model.fit(
        dataset_generators,
        dataset_steps_per_epoch,
        args.epochs,
        args.output_path,
        args.with_wandb,
    )

    # Save model
    model.save_model(args.output_path)

    print("Training complete!")


if __name__ == "__main__":
    main()
