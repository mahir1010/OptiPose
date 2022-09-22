import csv
import os

import numpy as np

from OptiPose import DLTrecon
from OptiPose import Skeleton, rotate
from OptiPose.data_store_interface import OptiPoseDataStore3D
from OptiPose.post_processor_interface.PostProcessorInterface import PostProcessorInterface


class ReconstructionProcess(PostProcessorInterface):
    PROCESS_NAME = "Reconstruction"

    def __init__(self, global_config, camera_config, data_readers, threshold, reconstruction_algorithm, scale,
                 file_prefix='Recon'):
        super(ReconstructionProcess, self).__init__(None)
        self.body_parts = global_config['body_parts']
        self.threshold = threshold
        self.data_readers = data_readers
        self.dlt_coefficients = np.array([camera['dlt_coefficients'] for camera in camera_config.values()])
        self.reconstruction_algorithm = reconstruction_algorithm
        self.rotation_matrix = np.array(global_config.get('OptiPose', {}).get('rotation_matrix', np.identity(3)),
                                        dtype=np.float)
        assert self.rotation_matrix.shape == (3, 3)
        self.scale = scale
        self.translation_matrix = np.array(global_config.get('OptiPose', {}).get('translation_matrix', [0, 0, 0]),
                                           dtype=np.float) * self.scale
        assert self.translation_matrix.shape == (3,)
        self.path = os.path.join(global_config['output_folder'],
                                 f"{file_prefix}_{self.threshold}_{'_'.join([camera for camera in camera_config]) if self.reconstruction_algorithm != 'auto_subset' else 'auto_subset'}.csv")

    def process(self, data_store):
        file = open(self.path, "w")
        csv_writer = csv.writer(file, delimiter=';')
        csv_writer.writerow(self.body_parts)
        length = len(min(self.data_readers, key=lambda x: len(x)))
        for iterator in range(length):
            self.progress = int(iterator / length * 100)
            skeleton_2D = [reader.get_skeleton(iterator) for reader in self.data_readers]
            recon_data = {}
            prob_data = {}
            for name in self.body_parts:
                subset = [sk[name] for sk in skeleton_2D]
                dlt_subset = self.dlt_coefficients
                if self.reconstruction_algorithm == "auto_subset":
                    indices = [subset[i].likelihood >= self.threshold for i in range(len(subset))]
                    if sum(indices) >= 2:
                        dlt_subset = dlt_subset[indices, :]
                        subset = [element for i, element in enumerate(subset) if indices[i]]
                recon_data[name] = rotate(DLTrecon(3, len(subset), dlt_subset, subset), self.rotation_matrix,
                                          self.scale) + self.translation_matrix
                prob_data[name] = min(subset, key=lambda x: x.likelihood).likelihood
            skeleton_3D = Skeleton(self.body_parts, recon_data, prob_data)
            csv_writer.writerow(
                [skeleton_3D[part].tolist() if skeleton_3D[part] > self.threshold else None for part in
                 self.body_parts])
        self.progress = 100
        self.data_ready = True

    def get_output(self):
        if self.data_ready:
            return OptiPoseDataStore3D(self.body_parts, self.path)
        else:
            return None
