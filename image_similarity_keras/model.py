import os
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

from typing import List, Tuple, Optional

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
        self.backbone_name = backbone_name
        self.backbone_hub_path = backbone_hub_path
        self.fc_depths = fc_depths
        self.emb_dims = emb_dims
        self.image_size = image_size

        self.model = None
        self.optimizer = None
        self.epochs = None

    def build(self, finetune: bool = False):
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
        assert self.model is not None, "Model is not built yet"

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
        # Check if model is built and compiled
        assert self.model is not None, "Model is not built yet"
        assert self.optimizer is not None, "Model is not compiled yet"

        # Callbacks
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                output_path, f"{self.backbone_name}_{self.batch_size}b_{epochs}ep"
            ),
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

        return train_history, evaluation_results

    def save_weights(self, output_path: str):
        assert self.model is not None, "Model is not built yet"

        self.model.save_weights(
            os.path.join(
                output_path,
                f"./model/{self.backbone_name}_{self.batch_size}b_{self.epochs}ep_final",
            )
        )
