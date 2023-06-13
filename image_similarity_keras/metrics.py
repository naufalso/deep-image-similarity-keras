import tensorflow as tf
import numpy as np


class TripletMetrics:
    def __init__(
        self,
        batch_size: int,
        min_thres: float = 0.3,
        max_thres: float = 0.81,
        interval: float = 0.1,
    ):
        self.batch_size = batch_size
        self.min_thres = min_thres
        self.max_thres = max_thres
        self.interval = interval

    def get_distance_metrics(self):
        return [
            DistanceOfPairs(self.batch_size, mode="max", pairs="pos"),
            DistanceOfPairs(self.batch_size, mode="min", pairs="neg"),
            DistanceOfPairs(self.batch_size, mode="avg", pairs="pos"),
            DistanceOfPairs(self.batch_size, mode="avg", pairs="neg"),
        ]

    def get_accuracy_metrics(self):
        return [
            AccOfPairs(self.batch_size, threshold=thres, pairs="pos")
            for thres in np.arange(self.min_thres, self.max_thres, self.interval)
        ]


class DistanceOfPairs(tf.keras.metrics.Metric):
    def __init__(self, batch_size, mode="max", pairs="pos", **kwargs):
        name = f"dist/{pairs}/{mode}"
        super(DistanceOfPairs, self).__init__(name=name, **kwargs)

        assert mode in ["max", "min", "avg"]
        assert pairs in ["pos", "neg"]

        aggregate_funcs = {
            "max": tf.math.reduce_max,
            "min": tf.math.reduce_min,
            "avg": tf.math.reduce_mean,
        }

        self.batch_size = batch_size
        self.aggregate_func = aggregate_funcs[mode]
        self.pair_distance_func = (
            self._positive_distance if pairs == "pos" else self._negative_distance
        )

        self.mean_distances = tf.keras.metrics.Mean(f"mean_{name}")

    def _positive_distance(self, grouped_embedding):
        # Positive pair distance shape: [BATCH_SIZE]
        positive_distances = tf.norm(
            grouped_embedding[:, 0, :] - grouped_embedding[:, 1, :], axis=1
        )

        aggregated_distance = self.aggregate_func(positive_distances)
        return aggregated_distance

    def _negative_distance(self, grouped_embedding):
        # Negative pair distance shape: [2, BATCH_SIZE-1]
        negative_distance_ori = tf.norm(
            grouped_embedding[:-1, 0, :] - grouped_embedding[1:, 0, :], axis=1
        )
        negative_distance_aug = tf.norm(
            grouped_embedding[:-1, 1, :] - grouped_embedding[1:, 1, :], axis=1
        )
        negative_distance = tf.stack([negative_distance_ori, negative_distance_aug])

        aggregated_distance = self.aggregate_func(negative_distance)
        return aggregated_distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = tf.convert_to_tensor(y_true, name="labels")
        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        # Group the same labels
        grouped_embedding = tf.stack(tf.split(embeddings, 2), axis=1)

        aggregated_distance = self.pair_distance_func(grouped_embedding)
        self.mean_distances.update_state(aggregated_distance)

    def result(self):
        return self.mean_distances.result()

    def reset_state(self):
        self.mean_distances.reset_state()


class AccOfPairs(tf.keras.metrics.Metric):
    def __init__(self, batch_size, threshold=0.65, pairs="pos", **kwargs):
        name = f"acc/{pairs}/{threshold:.2f}"
        super(AccOfPairs, self).__init__(name=name, **kwargs)

        assert pairs in ["pos", "neg"]

        self.batch_size = batch_size
        self.threshold = tf.constant(threshold, dtype=tf.float32)
        self.pair_acc_func = (
            self._positive_acc if pairs == "pos" else self._negative_acc
        )

        self.mean_acc = tf.keras.metrics.Mean(f"mean_{name}")

    def _positive_acc(self, grouped_embedding):
        # Positive pair distance shape: [BATCH_SIZE]
        positive_distances = tf.norm(
            grouped_embedding[:, 0, :] - grouped_embedding[:, 1, :], axis=1
        )
        correct_preds = tf.where(positive_distances <= self.threshold, 1.0, 0.0)

        return tf.reduce_mean(correct_preds)

    def _negative_acc(self, grouped_embedding):
        # Negative pair distance shape: [2, BATCH_SIZE-1]
        negative_distance_ori = tf.norm(
            grouped_embedding[:-1, 0, :] - grouped_embedding[1:, 0, :], axis=1
        )
        negative_distance_aug = tf.norm(
            grouped_embedding[:-1, 1, :] - grouped_embedding[1:, 1, :], axis=1
        )
        negative_distance = tf.stack([negative_distance_ori, negative_distance_aug])

        correct_preds = tf.where(negative_distance > self.threshold, 1.0, 0.0)

        return tf.reduce_mean(correct_preds)

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = tf.convert_to_tensor(y_true, name="labels")
        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        # Group the same labels
        grouped_embedding = tf.stack(tf.split(embeddings, 2), axis=1)

        aggregated_acc = self.pair_acc_func(grouped_embedding)
        self.mean_acc.update_state(aggregated_acc)

    def result(self):
        return self.mean_acc.result()

    def reset_state(self):
        self.mean_acc.reset_state()
