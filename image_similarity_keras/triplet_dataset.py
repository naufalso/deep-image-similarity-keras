import os

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import numpy as np

from typing import Tuple, Optional, Iterator


class TripletDataset:
    """
    Class for creating triplet dataset for training
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        target_size: Tuple[int, int],
    ):
        """
        Args:
            dataset_path (str): Path to the dataset folder.
            batch_size (int): Batch size.
            target_size (Tuple[int, int]): Target size of the images.
        """

        # Check if dataset_path contains train, val, and test folders
        if not all(
            [
                os.path.isdir(os.path.join(dataset_path, split))
                for split in ["train", "val", "test"]
            ]
        ):
            raise ValueError("dataset_path must contain train, val, and test folders.")

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.target_size = target_size

    def _triplet_pair_generator_function(
        self,
        anchor_gen: tf.keras.preprocessing.image.DirectoryIterator,
        positive_gen: tf.keras.preprocessing.image.DirectoryIterator,
    ) -> Iterator[Tuple[tf.Tensor, tf.Tensor]]:
        """
        Creates a generator function for triplet pairs.
        """
        while True:
            anchors, y_anc = next(anchor_gen)
            positives, y_pos = next(positive_gen)

            concatenate_img = tf.concat([anchors, positives], axis=0)
            concatenate_y = tf.concat([y_anc, y_pos], axis=0)

            yield concatenate_img, concatenate_y  # type: ignore

    def get_triplet_generator(
        self,
        rotation_range: int = 0,
        width_shift_range: float = 0.0,
        height_shift_range: float = 0.0,
        brightness_range: Optional[Tuple[float, float]] = None,
        shear_range: float = 0.0,
        zoom_range: float = 0.0,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        random_seed: Optional[int] = None,
    ) -> Tuple[
        Iterator[Tuple[tf.Tensor, tf.Tensor]],
        Iterator[Tuple[tf.Tensor, tf.Tensor]],
        Iterator[Tuple[tf.Tensor, tf.Tensor]],
    ]:
        """
        Creates triplet generators for training, validation, and test sets.

        Args:
            batch_size (int): Batch size.
            target_size (Tuple[int, int]): Target size of the images.
            rotation_range (int): Degree range for random rotations.
            width_shift_range (float): Fractional width shift range.
            height_shift_range (float): Fractional height shift range.
            brightness_range (Optional[Tuple[float, float]]): Brightness range.
            shear_range (float): Shear range.
            zoom_range (float): Zoom range.
            horizontal_flip (bool): Whether to perform random horizontal flips.
            vertical_flip (bool): Whether to perform random vertical flips.

        Returns:
            Tuple[Iterator[Tuple[tf.Tensor, tf.Tensor]], Iterator[Tuple[tf.Tensor, tf.Tensor]], Iterator[Tuple[tf.Tensor, tf.Tensor]]]: Tuple of triplet generators for training, validation, and test sets.
        """

        # Define data augmentators
        datagen_ori = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

        datagen_aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            brightness_range=brightness_range,
            rescale=1.0 / 255,
        )

        # Create random integer for seed
        if random_seed is None:
            seed = np.random.randint(0, 1000)
        else:
            seed = random_seed

        dataset_generator = []

        # Create triplet generators
        for split in ["train", "val", "test"]:
            # Create anchor generator
            anchor_generator = datagen_ori.flow_from_directory(
                os.path.join(self.dataset_path, split),
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode="sparse",
                seed=seed,
            )
            # Create positive generator
            pos_generator = datagen_aug.flow_from_directory(
                os.path.join(self.dataset_path, split),
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode="sparse",
                seed=seed,
            )

            # Create triplet generator
            triplet_generator = self._triplet_pair_generator_function(
                anchor_generator, pos_generator
            )

            dataset_generator.append(triplet_generator)

        return tuple(dataset_generator)
