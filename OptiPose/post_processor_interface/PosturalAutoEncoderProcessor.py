import numpy as np

from OptiPose.models.postural_autoencoder import optipose_postural_autoencoder
from OptiPose.models.utils import build_batch
from OptiPose.post_processor_interface.PostProcessorInterface import PostProcessorInterface


class PosturalAutoEncoderProcess(PostProcessorInterface):
    PROCESS_NAME = "Postural Auto-Encoder"

    def __init__(self, config, window_size, n_pcm, n_scm, n_heads, weights, overlap=0, output_dim=64,
                 translation_vector=np.array([0, 0, 0])):
        super(PosturalAutoEncoderProcess, self).__init__(None)
        self.window_size = window_size
        self.config = config
        self.model = optipose_postural_autoencoder(window_size, len(config['body_parts']), n_pcm, n_scm, n_heads,
                                                   output_dim, weights)
        self.overlap = max(overlap, 0)
        self.translation_vector = translation_vector

    def process(self, data_store, batch_size=40):
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
            model_output = self.model.predict(model_input, verbose=0)
            for sequence, indices in zip(model_output, batch_indices):
                for i, name in enumerate(self.config['body_parts']):
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
