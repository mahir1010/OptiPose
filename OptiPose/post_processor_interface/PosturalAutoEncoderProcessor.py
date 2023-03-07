import numpy as np

from OptiPose import OptiPoseConfig, MAGIC_NUMBER
from OptiPose.models.postural_autoencoder import optipose_postural_autoencoder
from OptiPose.models.utils import build_batch
from OptiPose.post_processor_interface import ClusterAnalysisProcess
from OptiPose.post_processor_interface.PostProcessorInterface import PostProcessor


class SequentialPosturalAutoEncoderProcess(PostProcessor):
    PROCESS_NAME = "Sequenctial Postural Auto-Encoder"

    def __init__(self, config: OptiPoseConfig, window_size, n_pcm, n_scm, n_heads, weights, overlap=0, output_dim=64,
                 translation_vector=np.array([0, 0, 0])):
        super(SequentialPosturalAutoEncoderProcess, self).__init__(None)
        self.window_size = window_size
        self.config = config
        self.model = optipose_postural_autoencoder(window_size, config.num_parts, n_pcm, n_scm, n_heads,
                                                   output_dim, weights)
        self.overlap = max(overlap, 0)
        self.translation_vector = np.array(translation_vector,dtype=np.float32)

    def process(self, data_store):
        batch_size = 1
        self.data_store = data_store
        index = 0
        self.data_ready = False
        self.progress = 0
        while index < len(data_store):
            self.progress = int(index / len(self.data_store) * 100)
            if self.PRINT and self.progress % 10 == 0:
                print(f'\r{self.progress}% complete', end='')
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
            translation_mask  = np.all(model_input!=[MAGIC_NUMBER,MAGIC_NUMBER,MAGIC_NUMBER],axis=-1)
            translation_matrix = np.zeros_like(model_input)
            translation_matrix[translation_mask]=self.translation_vector
            model_input += translation_matrix
            model_output = self.model.predict(model_input, verbose=0)-translation_matrix
            for sequence, indices in zip(model_output, batch_indices):
                for i, name in enumerate(self.config.body_parts):
                    data_store.set_keypoint_slice(indices, name, sequence[:indices[1] - indices[0], i, :])
            if end == len(data_store) - 1:
                break

        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None


class OccupancyPosturalAutoEncoderProcess(PostProcessor):
    PROCESS_NAME = "Occupancy-Based Postural Auto-Encoder"

    def __init__(self, config: OptiPoseConfig, window_size, n_pcm, n_scm, n_heads, weights, output_dim=64,
                 min_window=30,
                 translation_vector=np.array([0, 0, 0])):
        super(OccupancyPosturalAutoEncoderProcess, self).__init__(None)
        assert min_window < window_size
        self.window_size = window_size
        self.config = config
        self.model = optipose_postural_autoencoder(window_size, config.num_parts, n_pcm, n_scm, n_heads,
                                                   output_dim, weights)
        self.translation_vector = translation_vector
        self.min_window = min_window

    def process(self, data_store, max_batch_size=40):
        self.data_store = data_store
        if not self.data_store.verify_stats():
            if self.PRINT:
                print("Generating file statistics")
            cluster_analysis = ClusterAnalysisProcess()
            cluster_analysis.PRINT = self.PRINT
            cluster_analysis.process(self.data_store)
            self.data_store = cluster_analysis.get_output()
        self.data_ready = False
        self.progress = 0
        occupancy_ranges = [[0.7,1.0],[0.5,1.0]]
        current_occupancy_data = np.array(self.data_store.stats.occupancy_data)
        for occupancy_range in occupancy_ranges:
            if self.PRINT:
                print(f'Processing clusters with {occupancy_range[0]} <= Occupancy <= {occupancy_range[1]} ')
            clusters = self.data_store.stats.get_occupancy_clusters(*occupancy_range)
            total_length = len(clusters)
            cluster = None
            flag = True
            while flag:
                print(f'\r{len(clusters)}/{total_length}: {cluster}', end='')
                if cluster is None or cluster['end'] - cluster['begin'] < self.min_window:
                    if len(clusters) == 0:
                        break
                    cluster = clusters.pop(0)
                batch = []
                batch_indices = []
                for k in range(max_batch_size):
                    if cluster['end'] - cluster['begin'] < self.min_window:
                        break
                    begin = cluster['begin']
                    end = min(cluster['begin'] + self.window_size, cluster['end'])
                    current_occupancy_data[begin:end+1]=1.0
                    batch.append(np.array(build_batch(data_store, begin, end, self.window_size)))
                    batch_indices.append([begin, end])
                    cluster['begin'] = end
                if len(batch) > 0:
                    model_input = np.array(batch, dtype=np.float32)
                    model_output = self.model.predict(model_input, verbose=0)
                    for sequence, indices in zip(model_output, batch_indices):
                        for i, name in enumerate(self.config.body_parts):
                            data_store.set_keypoint_slice(indices, name, sequence[:indices[1] - indices[0], i, :])
                self.data_store.stats.occupancy_data = current_occupancy_data.tolist()
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None
