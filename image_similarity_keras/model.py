# Author: Naufal Suryanto (https://github.com/naufalso)

import os
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

from typing import List

from .metrics import TripletMetrics


class SiameseModel:
    def __init__(
        self,
        backbone_name: str,
        backbone_hub_path: str,
        fc_depths: List[int] = [512, 256],
        emb_dims: int = 128,
        image_size: int = 224,
    ):
        """
        Initializes a SiameseModel object with the given parameters.

        Args:
            backbone_name (str): Name of the backbone model to use.
            backbone_hub_path (str): Path to the TensorFlow Hub module for the backbone model.
            fc_depths (List[int], optional): List of fully connected layer depths. Defaults to [512, 256].
            emb_dims (int, optional): Dimensionality of the final embedding output. Defaults to 128.
            image_size (int, optional): Size of the input images. Defaults to 224.
        """

        self.backbone_name = backbone_name
        self.backbone_hub_path = backbone_hub_path
        self.fc_depths = fc_depths
        self.emb_dims = emb_dims
        self.image_size = image_size

        self.finetune = False
        self.batch_size = None

        self.model = None
        self.optimizer = None
        self.epochs = None

    def build(self, finetune: bool = False):
        """
        Builds the SiameseModel object with the given parameters.

        Args:
            finetune (bool, optional): Whether to fine-tune the model. Defaults to False.

        Returns:
            SiameseModel: The built SiameseModel object.
        """
        # Set finetune flag
        self.finetune = finetune

        # Build a model
        tf.keras.backend.clear_session()
        siamese_model = tf.keras.Sequential(name=f"siamese_{self.backbone_name}")

        # Add pretrained model as the feature extractors (Weights frozen)
        siamese_model.add(
            hub.KerasLayer(
                self.backbone_hub_path, trainable=finetune, name=self.backbone_name
            )
        )

        # Add fully connected layers
        for dense_depth in self.fc_depths:
            siamese_model.add(tf.keras.layers.Dense(dense_depth, activation="relu"))

        # Add last fully connected layers as final embedding output without activation
        siamese_model.add(
            tf.keras.layers.Dense(self.emb_dims, activation=None, name="out_emb")
        )

        # Add lambda function for L2 normalize embeddings
        siamese_model.add(
            tf.keras.layers.Lambda(
                lambda x: tf.math.l2_normalize(x, axis=1), name="l2_norm"
            )
        )

        # Build the model
        siamese_model.build([None, self.image_size, self.image_size, 3])
        siamese_model.summary()

        self.model = siamese_model

        return self

    def compile(
        self,
        batch_size: int,
        optimizer: str,
        learning_rate: float,
        loss: str,
        metrics_config: dict,
    ):
        """
        Compiles the model with the given parameters.

        Args:
            batch_size (int): Batch size.
            optimizer (str): Name of the optimizer to use.
            learning_rate (float): Learning rate.
            loss (str): Name of the loss function to use.
            metrics_config (dict): Configuration for the metrics.

        Returns:
            SiameseModel: The compiled SiameseModel object.
        """

        assert self.model is not None, "Model is not built yet"

        # Update model parameters
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.optimizer.lr.assign(learning_rate)  # type: ignore

        self.triplet_metrics = TripletMetrics(batch_size, **metrics_config)
        self.metrics = [
            *self.triplet_metrics.get_distance_metrics(),
            *self.triplet_metrics.get_accuracy_metrics(),
        ]

        if loss == "triplet":
            self.loss = tfa.losses.TripletSemiHardLoss()
        else:
            raise ValueError(f"Loss {loss} is not supported.")

        # Compile the model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        return self

    def fit(
        self,
        dataset_generators,
        dataset_steps_per_epoch,
        epochs,
        output_path,
        with_wandb=True,
    ):
        """
        Fits the model with the given parameters.

        Args:
            dataset_generators ([type]): [description]
            dataset_steps_per_epoch ([type]): [description]
            epochs ([type]): [description]
            output_path ([type]): [description]
            with_wandb (bool, optional): [description]. Defaults to True.

        Returns:
            train_history: keras.callbacks.History
            evaluation_results: List[float]
        """

        # Check if model is built and compiled
        assert self.model is not None, "Model is not built yet"
        assert self.optimizer is not None, "Model is not compiled yet"

        # Callbacks
        model_name = f"{self.backbone_name}_{self.batch_size}b_{epochs}ep"
        model_path = os.path.join(output_path, "best_val", model_name)
        os.makedirs(model_path, exist_ok=True)

        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_path, self.backbone_name),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            save_freq="epoch",
        )

        self.callbacks = [self.model_checkpoint_callback]
        # If use wandb, add wandb callback
        if with_wandb:
            from wandb.keras import WandbMetricsLogger

            self.callbacks.append(WandbMetricsLogger())

        train_generator, valid_generator, test_generator = dataset_generators
        (
            train_step_per_epoch,
            val_step_per_epoch,
            test_step_per_epoch,
        ) = dataset_steps_per_epoch

        self.epochs = epochs

        train_history = self.model.fit(
            train_generator,
            steps_per_epoch=train_step_per_epoch,
            validation_data=valid_generator,
            validation_steps=val_step_per_epoch,
            epochs=epochs,
            callbacks=self.callbacks,
        )

        evaluation_results = self.model.evaluate(
            test_generator, steps=test_step_per_epoch
        )

        if with_wandb:
            import wandb

            for metric_name, metric_value in zip(
                self.model.metrics_names, evaluation_results
            ):
                wandb.run.summary[f"test/{metric_name}"] = metric_value

        return train_history, evaluation_results

    def to_json(self):
        """
        Returns the model configuration as a dictionary.

        Returns:
            model_config: dict
        """
        model_config = {
            "backbone_name": self.backbone_name,
            "backbone_hub_path": self.backbone_hub_path,
            "fc_depths": self.fc_depths,
            "emb_dims": self.emb_dims,
        }

        if self.model is not None:
            training_config = {
                "finetune": self.finetune,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "optimizer": self.optimizer.__class__.__name__,
                "learning_rate": self.optimizer.lr.numpy(),
                "loss": self.loss.__class__.__name__,
            }
            model_config["training_config"] = training_config

        return model_config

    def save_model(self, output_path: str):
        """
        Saves the model architecture and weights to the given path.

        Args:
            output_path (str): Path to save the model.
        """
        assert self.model is not None, "Model is not built yet"

        model_name = f"{self.backbone_name}_{self.batch_size}b_{self.epochs}ep_final"
        model_path = os.path.join(output_path, "model", model_name)
        os.makedirs(model_path, exist_ok=True)

        # Save model architecture
        model_json = self.to_json()

        with open(
            os.path.join(model_path, f"{self.backbone_name}.json"), "w"
        ) as json_file:
            json_file.write(model_json)

        # Save model weights
        self.model.save_weights(os.path.join(model_path, self.backbone_name + ".h5"))
