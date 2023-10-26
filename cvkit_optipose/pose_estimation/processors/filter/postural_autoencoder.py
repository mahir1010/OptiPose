import numpy as np
from cvkit import MAGIC_NUMBER
from cvkit.pose_estimation.config import PoseEstimationConfig
from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData
from cvkit.pose_estimation.processors.util import ClusterAnalysis
from OptiPose.model.postural_autoencoder import optipose_postural_autoencoder
from OptiPose.model.utils import build_batch
import tensorflow as tf


class SequentialPosturalAutoEncoder(Processor):

    PROCESSOR_NAME = "Sequential Postural Auto-Encoder"
    PROCESSOR_ID = "optipose_sequential"
    META_DATA = {
        'config': ProcessorMetaData('config', ProcessorMetaData.GLOBAL_CONFIG),
        'window_size': ProcessorMetaData('Temporal Window', ProcessorMetaData.INT, min_val=1),
        'n_pcm': ProcessorMetaData('Parallel Context Models', ProcessorMetaData.INT, min_val=1),
        'n_scm': ProcessorMetaData('Sub-Context Models', ProcessorMetaData.INT, min_val=1),
        'n_heads': ProcessorMetaData('Heads', ProcessorMetaData.INT, min_val=1),
        'weights': ProcessorMetaData('Weights Directory', ProcessorMetaData.DIR_PATH),
        'overlap': ProcessorMetaData('Overlap Window', ProcessorMetaData.INT, default=0, min_val=0),
        'output_dim': ProcessorMetaData('Output Embedding Dimension', ProcessorMetaData.INT, min_val=1)}

    def __init__(self, config: PoseEstimationConfig, window_size, n_pcm, n_scm, n_heads, weights, overlap=0,
                 output_dim=64,
                 translation_vector=np.array([0, 0, 0])):
        super(SequentialPosturalAutoEncoder, self).__init__()
        self.window_size = window_size
        self.config = config
        self.n_pcm = n_pcm
        self.n_scm = n_scm
        self.n_heads = n_heads
        self.weights = weights
        self.overlap = max(overlap, 0)
        self.translation_vector = np.array(translation_vector, dtype=np.float32)
        self.model = None

    def process(self, data_store):

        batch_size = 1
        self._data_store = data_store
        index = 0
        self._data_ready = False
        self._progress = 0
        tf.keras.backend.clear_session()
        self.model = optipose_postural_autoencoder(self.window_size, self.config.num_parts, self.n_pcm, self.n_scm, self.n_heads,
                                                   self.weights)
        while index < len(data_store):
            self._progress = int(index / len(self._data_store) * 100)
            if self.PRINT and self._progress % 10 == 0:
                print(f'\r{self._progress}% complete', end='')
            batch = []
            batch_indices = []
            for k in range(batch_size):
                begin = index
                end = min(index + self.window_size, len(data_store) - 1)
                batch.append(np.array(build_batch(data_store, begin, end, self.window_size)))
                batch_indices.append([begin, end])
                index = end - self.overlap
                if end == len(data_store) - 1:
                    break
            model_input = np.array(batch, dtype=np.float32)
            translation_mask = np.all(model_input != [MAGIC_NUMBER, MAGIC_NUMBER, MAGIC_NUMBER], axis=-1)
            translation_matrix = np.zeros_like(model_input)
            translation_matrix[translation_mask] = self.translation_vector
            model_input += translation_matrix
            model_output = self.model.predict(model_input, verbose=0) - translation_matrix
            for sequence, indices in zip(model_output, batch_indices):
                for i, name in enumerate(self.config.body_parts):
                    data_store.set_part_slice(indices, name, sequence[:indices[1] - indices[0], i, :])
            if end == len(data_store) - 1:
                break
        del self.model
        tf.keras.backend.clear_session()
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None


class OccupancyPosturalAutoEncoder(Processor):
    PROCESSOR_NAME = "Occupancy-Based Postural Auto-Encoder"
    PROCESSOR_ID = "optipose_occupancy"
    META_DATA = {
        'config': ProcessorMetaData('config', ProcessorMetaData.GLOBAL_CONFIG),
        'window_size': ProcessorMetaData('Temporal Window', ProcessorMetaData.INT, min_val=1),
        'n_pcm': ProcessorMetaData('Parallel Context Models', ProcessorMetaData.INT, min_val=1),
        'n_scm': ProcessorMetaData('Sub-Context Models', ProcessorMetaData.INT, min_val=1),
        'n_heads': ProcessorMetaData('Heads', ProcessorMetaData.INT, min_val=1),
        'max_batch_size': ProcessorMetaData('Max Batch Size', ProcessorMetaData.INT, min_val=1,default=40),
        'weights': ProcessorMetaData('Weights Directory', ProcessorMetaData.DIR_PATH),
        'min_window': ProcessorMetaData('Overlap Window', ProcessorMetaData.INT, default=30, min_val=1)}
    REQUIRES_STATS = True

    def __init__(self, config: PoseEstimationConfig, window_size, n_pcm, n_scm, n_heads, weights,
                 min_window=30,max_batch_size=40,
                 translation_vector=np.array([0, 0, 0])):
        super(OccupancyPosturalAutoEncoder, self).__init__()
        assert min_window < window_size
        self.window_size = window_size
        self.config = config
        self.n_pcm = n_pcm
        self.n_scm = n_scm
        self.n_heads = n_heads
        self.weights = weights
        self.translation_vector = translation_vector
        self.min_window = min_window
        self.max_batch_size = max_batch_size
        self.model = None

    def process(self, data_store):
        self._data_store = data_store
        if not self._data_store.verify_stats():
            if self.PRINT:
                print("Generating file statistics")
            cluster_analysis = ClusterAnalysis()
            cluster_analysis.PRINT = self.PRINT
            cluster_analysis.process(self._data_store)
            self._data_store = cluster_analysis.get_output()
        self._data_ready = False
        self._progress = 0
        tf.keras.backend.clear_session()
        self.model = optipose_postural_autoencoder(self.window_size, self.config.num_parts, self.n_pcm, self.n_scm,
                                                   self.n_heads,
                                                   self.weights)
        occupancy_ranges = [[0.7, 1.0], [0.5, 1.0], [0.3,1.0]]
        current_occupancy_data = np.array(self._data_store.stats.occupancy_data)
        for occupancy_range in occupancy_ranges:
            if self.PRINT:
                print(f'Processing clusters with {occupancy_range[0]} <= Occupancy <= {occupancy_range[1]} ')
            clusters = self._data_store.stats.get_occupancy_clusters(*occupancy_range)
            total_length = len(clusters)
            cluster = None
            flag = True
            while flag:
                if cluster is None or cluster['end'] - cluster['begin'] < self.min_window:
                    if len(clusters) == 0:
                        break
                    cluster = clusters.pop(0)
                print(f'\r{len(clusters)}/{total_length}: {cluster}', end='')
                batch = []
                batch_indices = []
                for k in range(self.max_batch_size):
                    if cluster['end'] - cluster['begin'] < self.min_window:
                        break
                    begin = cluster['begin']
                    end = min(cluster['begin'] + self.window_size, cluster['end'])
                    current_occupancy_data[begin:end + 1] = 1.0
                    batch.append(np.array(build_batch(data_store, begin, end, self.window_size)))
                    batch_indices.append([begin, end])
                    cluster['begin'] = end
                if len(batch) > 0:
                    model_input = np.array(batch, dtype=np.float32)
                    model_output = self.model.predict(model_input, verbose=0)
                    for sequence, indices in zip(model_output, batch_indices):
                        for i, name in enumerate(self.config.body_parts):
                            data_store.set_part_slice(indices, name, sequence[:indices[1] - indices[0], i, :])
                self._data_store.stats.occupancy_data = current_occupancy_data.tolist()
        del self.model
        tf.keras.backend.clear_session()
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None
