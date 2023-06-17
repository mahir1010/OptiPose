import csv
from random import uniform

import numpy as np
import tensorflow as tf
from cvkit import MAGIC_NUMBER
from cvkit.pose_estimation.data_readers import DataStoreInterface
from cvkit.pose_estimation.processors.util import ClusterAnalysis


@tf.function
def distance_map(ref, points):
    return tf.vectorized_map(fn=lambda t: tf.keras.backend.sqrt(1e-9 + tf.reduce_sum(tf.keras.backend.square(ref - t))),
                             elems=points)


@tf.function
def distance_map_non_zero(ref, points):
    return tf.vectorized_map(
        fn=lambda t: tf.keras.backend.sqrt(1e-9 + tf.reduce_sum(tf.keras.backend.abs(ref))) + tf.keras.backend.sqrt(
            1e-9 + tf.reduce_sum(tf.keras.backend.square(t - ref))), elems=points)


def build_spatio_temporal_loss(spatial_factor=0.0001, temporal_factor=0.0001):
    assert spatial_factor <= 1 and temporal_factor <= 1
    huber = tf.keras.losses.Huber()

    @tf.function
    def loss_fn(y_t_o, y_p):
        y_t = tf.cast(y_t_o, dtype=tf.float32)
        y_p = tf.cast(y_p, dtype=tf.float32)
        mask = tf.cast(tf.logical_not(tf.reduce_all(tf.reduce_all(y_t == 0, axis=-1), axis=-1)),
                       dtype=tf.float32)  # bx30
        total = tf.reduce_sum(mask, axis=-1)
        huber_loss = huber(y_t, y_p)
        temp = y_p[:, 1:, :, :]
        temp1 = y_t[:, 1:, :, :]
        v_t = tf.vectorized_map(
            lambda row: tf.vectorized_map(lambda y: tf.vectorized_map(lambda x: distance_map(x, y), elems=y),
                                          elems=row), elems=y_t)
        v_p = tf.vectorized_map(
            lambda row: tf.vectorized_map(lambda y: tf.vectorized_map(lambda x: distance_map(x, y), elems=y),
                                          elems=row), elems=y_p)
        spatial_loss = tf.reduce_mean(
            tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.abs(v_t - v_p), axis=-1), axis=-1) * mask,
                          axis=-1) / total)

        mask = mask[:, 1:]
        temporal_loss = tf.reduce_mean(tf.reduce_sum(
            tf.reduce_mean(tf.reduce_sum(tf.abs((temp1 - y_t[:, :-1, :, :]) - (temp - y_p[:, :-1, :, :])), axis=-2),
                           axis=-1) * mask, axis=-1) / total)
        return huber_loss + (spatial_factor * spatial_loss) + (temporal_factor * temporal_loss)

    return loss_fn


def evaluate_predictions(prediction: DataStoreInterface, ground_truth: DataStoreInterface, metric_fns, sequence_len=60,
                         batch_size=60, index_limit=None, verbose=False):
    assert prediction.DIMENSIONS == ground_truth.DIMENSIONS

    if not ground_truth.verify_stats():
        analysis_process = ClusterAnalysis()
        analysis_process.PRINT = True
        analysis_process.process(ground_truth)
        ground_truth = analysis_process.get_output()
    batch_results = {}
    index_limit = min(len(prediction), len(ground_truth)) if index_limit is None else index_limit
    for i in range(len(metric_fns)):
        batch_results[metric_fns[i]._name] = []
    pred_batch = []
    gt_batch = []
    print_count = 0
    total_counts = len(ground_truth) if index_limit is None else index_limit
    PAD = [[0, 0, 0]] * len(ground_truth.body_parts)
    for accurate_data_point in ground_truth.stats.iter_accurate_clusters():
        index = accurate_data_point['begin']
        if index_limit is not None and index > index_limit:
            break
        while index < accurate_data_point['end']:
            if print_count % 100 == 0 and verbose:
                print(f"{index}/{total_counts}")
            ground_truth_sequence = []
            prediction_sequence = []
            for iterator in range(index, min(index + sequence_len, accurate_data_point['end'] + 1)):
                ground_truth_sequence.append(ground_truth.get_numpy(iterator))
                prediction_sequence.append(prediction.get_numpy(iterator))
            index = iterator
            while len(prediction_sequence) != sequence_len:
                prediction_sequence.append(PAD.copy())
                ground_truth_sequence.append(PAD.copy())
            prediction_sequence = np.array(prediction_sequence, dtype=np.float)
            ground_truth_sequence = np.array(ground_truth_sequence, dtype=np.float)
            pred_batch.append(prediction_sequence)
            gt_batch.append(ground_truth_sequence)
            if len(pred_batch) == batch_size:
                for metric_index, metric_fn in enumerate(metric_fns):
                    batch_results[metric_fns[metric_index]._name].append(
                        metric_fn(np.stack(gt_batch, axis=0), np.stack(pred_batch, axis=0)).numpy())
                pred_batch.clear()
                gt_batch.clear()
            print_count += 1
    if len(pred_batch) > 0:
        for metric_index, metric_fn in enumerate(metric_fns):
            batch_results[metric_fns[metric_index]._name].append(
                metric_fn(np.stack(gt_batch, axis=0), np.stack(pred_batch, axis=0)).numpy())
            batch_results[metric_fns[metric_index]._name] = np.mean(batch_results[metric_fns[metric_index]._name],
                                                                    axis=0).round(2)
    return batch_results


def build_batch(data_store: DataStoreInterface, begin, end, max_seq_length, is_dataset=False, increment=1,
                truncate=False):
    batch = []
    BLANK = [MAGIC_NUMBER] * 3
    PAD = np.zeros((len(data_store.body_parts), 3))
    for i in range(begin, end, increment):
        if len(batch) >= max_seq_length:
            break
        batch.append(data_store.get_numpy(i))
        if is_dataset:
            assert not np.any(np.all(batch[-1] == BLANK, axis=1))
            if truncate:
                batch[-1] = batch[-1].astype(np.int)
        batch[-1] = batch[-1].tolist()
    if not is_dataset:
        while len(batch) < max_seq_length:
            batch.append(PAD)
    return batch


def train(model, train_dataset, test_dataset, loss_function, epochs, lr=5e-4, metrics=["mae"], callbacks=[],
          clipnorm=1.0):
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(loss=loss_function, optimizer=opt, metrics=metrics)
    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=2,
                        callbacks=callbacks)
    return history


def mask_random_keypoint(source: DataStoreInterface, pose_level_probability=0.5, marker_level_probability=0.5):
    assert pose_level_probability <= 0.5 and marker_level_probability <= 0.5
    writer = csv.writer(open(f'{source.base_file_path}_masked.csv', 'w'), delimiter=source.SEP)
    writer.writerows(source.get_header_rows())
    empty_skeleton = source.build_empty_skeleton()
    body_parts = source.body_parts
    for index, sk in source.row_iterator():
        if index % 500 == 0:
            print(f'\r{index}/{len(source)}', end='')
        if uniform(0, 1) <= pose_level_probability:
            for i in range(len(body_parts)):
                if uniform(0, 1) <= marker_level_probability:
                    sk[body_parts[i]] = empty_skeleton[body_parts[i]]
        sk = source.convert_to_list(index, sk)
        writer.writerow(sk)
