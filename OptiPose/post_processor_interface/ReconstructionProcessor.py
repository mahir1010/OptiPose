import numpy as np

from OptiPose import DLTrecon, OptiPoseConfig
from OptiPose import Skeleton, rotate
from OptiPose.data_store_interface import OptiPoseDataStore3D
from OptiPose.post_processor_interface.PostProcessorInterface import PostProcessor


class ReconstructionProcess(PostProcessor):
    PROCESS_NAME = "Reconstruction"

    def __init__(self, global_config: OptiPoseConfig, source_views, data_readers, threshold, reconstruction_algorithm,
                 scale):
        super(ReconstructionProcess, self).__init__(None)
        self.body_parts = global_config.body_parts
        self.threshold = threshold
        self.data_readers = data_readers
        self.dlt_coefficients = np.array([global_config.views[view].dlt_coefficients for view in source_views])
        self.reconstruction_algorithm = reconstruction_algorithm
        self.rotation_matrix = np.array(global_config.rotation_matrix)
        assert self.rotation_matrix.shape == (3, 3)
        self.scale = scale
        self.translation_matrix = np.array(global_config.translation_matrix) * self.scale
        assert self.translation_matrix.shape == (3,)

    def process(self, data_store):
        self.out_csv = OptiPoseDataStore3D(self.body_parts, None)
        length = len(min(self.data_readers, key=lambda x: len(x)))
        self.data_ready = False
        self.progress = 0
        for iterator in range(length):
            self.progress = int(iterator / length * 100)
            skeleton_2D = [reader.get_skeleton(iterator) for reader in self.data_readers]
            recon_data = {}
            prob_data = {}
            for name in self.body_parts:
                subset = [sk[name] for sk in skeleton_2D]
                dlt_subset = self.dlt_coefficients
                indices = [subset[i].likelihood >= self.threshold for i in range(len(subset))]
                if (self.reconstruction_algorithm == "auto_subset" and sum(indices) >= 2) or sum(indices) == len(
                        self.data_readers):
                    dlt_subset = dlt_subset[indices, :]
                    subset = [element for i, element in enumerate(subset) if indices[i]]
                    recon_data[name] = rotate(DLTrecon(3, len(subset), dlt_subset, subset), self.rotation_matrix,
                                              self.scale) + self.translation_matrix
                    prob_data[name] = min(subset, key=lambda x: x.likelihood).likelihood
            skeleton_3D = Skeleton(self.body_parts, recon_data, prob_data)
            self.out_csv.set_skeleton(iterator, skeleton_3D)
        self.progress = 100
        self.data_ready = True

    def get_output(self):
        if self.data_ready:
            # return OptiPoseDataStore3D(self.body_parts, self.path)
            return self.out_csv
        else:
            return None
