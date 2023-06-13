import os
import argparse
import wandb
import json

from image_similarity_keras.triplet_dataset import TripletDataset
from image_similarity_keras.model import SiameseModel


def main():
    parser = argparse.ArgumentParser(
        description="Argument parser for image similarity project"
    )

    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/models/convnext_small.json",
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
        "--loss", type=str, default="triplet", help="Loss function to use for training"
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
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="image_similarity",
        help="Wandb project name",
    )

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
    wandb.init(
        project=args.wandb_project,
        name=f"{model_config_ns.backbone_name}_{args.batch_size}b_{args.epochs}ep",
        config={
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
    history, evaluation = model.fit(
        dataset_generators,
        dataset_steps_per_epoch,
        args.epochs,
        args.output_path,
        args.wandb_project,
    )

    # Save model
    model.save_weights(args.output_path)

    # TODO: Save history and evaluation

    print(evaluation)
    print("Training complete!")


if __name__ == "__main__":
    main()
