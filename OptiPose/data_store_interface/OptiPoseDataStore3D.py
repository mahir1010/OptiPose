import os

import numpy as np
import pandas as pd

from OptiPose import MAGIC_NUMBER
from OptiPose import convert_to_numpy
from OptiPose.data_store_interface.DataStoreInterface import DataStoreInterface
from OptiPose.skeleton import Skeleton, Part


class OptiPoseDataStore3D(DataStoreInterface):
    FLAVOR = "OptiPose3D"
    SEP = ';'

    def save_file(self, path: str = None) -> None:
        if path is None:
            path = self.path
        self.data.sort_index(inplace=True)
        self.data.to_csv(path, index=False, sep=';')

    def delete_marker(self, index, name, force_remove=False):
        if force_remove or index in self.data.index:
            self.data.loc[index, name] = pd.NA

    def set_behaviour(self, index, behaviour: str) -> None:
        self.data[index, 'behaviour'] = behaviour

    def get_behaviour(self, index) -> str:
        if index in self.data.index:
            return self.data.loc[index, 'behaviour']
        else:
            return ""

    def get_keypoint_slice(self, slice_indices: list, name: str) -> np.ndarray:
        return self.data.loc[slice_indices[0]:slice_indices[1], name].map(lambda x: self.build_part(x, name)).to_numpy()

    def set_keypoint_slice(self, slice_indices: list, name: str, data: np.ndarray) -> None:
        self.data.loc[slice_indices[0]:slice_indices[1], name] = data[:]

    def get_marker(self, index, name) -> Part:
        if index in self.data.index:
            pt = convert_to_numpy(self.data.loc[index, name])
            return Part(pt, name, float(not all(pt == MAGIC_NUMBER)))
        else:
            return Part([MAGIC_NUMBER] * self.DIMENSIONS, name, 0.0)

    def set_marker(self, index, part: Part) -> None:
        name = part.name
        self.data.loc[index, name] = str(part.tolist())
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def build_skeleton(self, row) -> Skeleton:
        part_map = {}
        likelihood_map = {}
        for name in self.body_parts:
            part_map[name] = convert_to_numpy(row[name])
            likelihood_map[name] = float(not all(part_map[name] == MAGIC_NUMBER))
        return Skeleton(self.body_parts, part_map=part_map, likelihood_map=likelihood_map, behaviour=row['behaviour'])

    def build_part(self, arr, name):
        pt = convert_to_numpy(arr)
        return Part(pt, name, float(not all(pt == MAGIC_NUMBER)))

    def __init__(self, body_parts, path):
        super(OptiPoseDataStore3D, self).__init__(body_parts, path)
        self.path = path
        if os.path.exists(path):
            self.data = pd.read_csv(path, sep=';')
        else:
            self.data = pd.DataFrame(columns=body_parts)
        for part in body_parts:
            if part not in self.data.columns:
                self.data[part] = ""

        if "behaviour" not in self.data.columns:
            self.data['behaviour'] = ""
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    @staticmethod
    def convert_to_list(index, skeleton, threshold=0.8):
        return [skeleton[part].tolist() if skeleton[part] > threshold else None for part in skeleton.body_parts]
