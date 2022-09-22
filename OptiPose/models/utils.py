import numpy as np
import tensorflow as tf

from OptiPose.data_store_interface import DataStoreInterface
from OptiPose.post_processor_interface import ClusterAnalysisProcess


@tf.function
def distance_map(ref, points):
    return tf.vectorized_map(fn=lambda t: tf.keras.backend.sqrt(1e-9 + tf.reduce_sum(tf.keras.backend.square(ref - t))),
                             elems=points)


@tf.function
def distance_map_non_zero(ref, points):
    return tf.vectorized_map(
        fn=lambda t: tf.keras.backend.sqrt(1e-9 + tf.reduce_sum(tf.keras.backend.abs(ref))) + tf.keras.backend.sqrt(
            1e-9 + tf.reduce_sum(tf.keras.backend.square(t - ref))), elems=points)


def evaluate_predictions(prediction: DataStoreInterface, ground_truth: DataStoreInterface, metric_fns, sequence_len=60,
                         batch_size=60):
    assert prediction.DIMENSIONS == ground_truth.DIMENSIONS
    if not ground_truth.verify_stats():
        analysis_process = ClusterAnalysisProcess()
        analysis_process.PRINT = True
        analysis_process.process(ground_truth)
        ground_truth = analysis_process.get_output()
    batch_results = {}
    for i in range(len(metric_fns)):
        batch_results[i] = []
    pred_batch = []
    gt_batch = []
    print_count = 0
    PAD = [[0, 0, 0]] * len(ground_truth.body_parts)
    for accurate_data_point in ground_truth.stats.iter_accurate_clusters():
        index = accurate_data_point['begin']
        while index < accurate_data_point['end']:
            if print_count % 100 == 0:
                print(f"{index}/{len(ground_truth)}")
            ground_truth_sequence = []
            prediction_sequence = []
            for iterator in range(index, min(index + sequence_len, accurate_data_point['end'] + 1)):
                ground_truth_sequence.append(ground_truth.get_numpy(iterator))
                prediction_sequence.append(prediction.get_numpy(iterator))
            index = iterator
            while len(prediction_sequence) != sequence_len:
                prediction_sequence.append(PAD.copy())
                ground_truth_sequence.append(PAD.copy())
            prediction_sequence = np.expand_dims(np.array(prediction_sequence, dtype=np.float), 0)
            ground_truth_sequence = np.expand_dims(np.array(ground_truth_sequence, dtype=np.float), 0)
            pred_batch.append(prediction_sequence)
            gt_batch.append(ground_truth_sequence)
            if len(pred_batch) == batch_size:
                for metric_fn_id, metric_fn in enumerate(metric_fns):
                    batch_results[metric_fn_id].append(
                        metric_fn(np.stack(gt_batch, axis=0), np.stack(pred_batch, axis=0)).numpy())
                pred_batch.clear()
                gt_batch.clear()
            print_count += 1
    if len(pred_batch) > 0:
        for metric_fn_id, metric_fn in enumerate(metric_fns):
            batch_results[metric_fn_id].append(
                metric_fn(np.stack(gt_batch, axis=0), np.stack(pred_batch, axis=0)).numpy())
            batch_results[metric_fn_id] = np.mean(batch_results[metric_fn_id], axis=0)
    return batch_results
