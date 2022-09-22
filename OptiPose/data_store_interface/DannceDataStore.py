import os

import numpy as np
import pandas as pd

from OptiPose import MAGIC_NUMBER
from OptiPose.data_store_interface.DataStoreInterface import DataStoreInterface
from OptiPose.skeleton import Skeleton, Part


class DannceDataStore(DataStoreInterface):
    FLAVOR = "dannce"
    DIMENSIONS = 3

    def __init__(self, body_parts, path):
        """
        Plugin to support DANNCE data file, converted from the matlab data file
        Args:
            body_parts: list of column names
            path: path to file
        """
        super(DannceDataStore, self).__init__(body_parts, path)
        self.path = path
        if os.path.exists(path):
            self.data = pd.read_csv(path, sep=',')
        else:
            self.data = pd.DataFrame(columns=body_parts)
        for part in body_parts:
            for dim in range(1, self.DIMENSIONS + 1):
                if f"{part}_{dim}" not in self.data.columns:
                    # self.data[part] = f"[{MAGIC_NUMBER},{MAGIC_NUMBER},{MAGIC_NUMBER}]"
                    self.data[f"{part}_{dim}"] = ""

        if "behaviour" not in self.data.columns:
            self.data['behaviour'] = ""
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def save_file(self, path: str = None) -> None:
        if path is None:
            path = self.path
        self.data.sort_index(inplace=True)
        self.data.to_csv(path, index=False, sep=';')

    def delete_marker(self, index, name, force_remove=False):
        if force_remove or index in self.data.index:
            self.data.loc[index, [f"{name}_{i}" for i in range(1, self.DIMENSIONS + 1)]] = pd.NA

    def set_behaviour(self, index, behaviour: str) -> None:
        self.data[index, 'behaviour'] = behaviour

    def get_behaviour(self, index) -> str:
        if index in self.data.index:
            return self.data.loc[index, 'behaviour']
        else:
            return ""

    def get_keypoint_slice(self, slice_indices: list, name: str) -> np.ndarray:
        return self.data.loc[slice_indices[0]:slice_indices[1],
               [f"{name}_{i}" for i in range(1, self.DIMENSIONS + 1)]].apply(
            lambda x: self.build_part(x, name), axis=1).to_numpy()

    def set_keypoint_slice(self, slice_indices: list, name: str, data: np.ndarray) -> None:
        for i in range(1, self.DIMENSIONS + 1):
            self.data.loc[slice_indices[0]:slice_indices[1], f"{name}_{i}"] = [d[i - 1] for d in data]

    def get_marker(self, index, name) -> Part:
        if index in self.data.index:
            pt = np.array([self.data.loc[index, f"{name}_{i}"] for i in range(1, self.DIMENSIONS + 1)])
            if any(np.isnan(pt)):
                pt = np.array([MAGIC_NUMBER] * self.DIMENSIONS)
            return Part(pt, name, float(not all(pt == MAGIC_NUMBER)))
        else:
            return Part([MAGIC_NUMBER] * self.DIMENSIONS, name, 0.0)

    def set_marker(self, index, part: Part) -> None:
        name = part.name
        for i in range(1, part.shape[0] + 1):
            self.data.loc[index, f"{name}_{i}"] = part[i - 1]
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def build_skeleton(self, row) -> Skeleton:
        part_map = {}
        likelihood_map = {}
        for name in self.body_parts:
            part_map[name] = np.array([row[f"{name}_{i}"] for i in range(1, self.DIMENSIONS + 1)])
            if any(np.isnan(part_map[name])):
                part_map[name] = np.array([MAGIC_NUMBER] * self.DIMENSIONS)
            likelihood_map[name] = float(not all(part_map[name] == MAGIC_NUMBER))
        return Skeleton(self.body_parts, part_map=part_map, likelihood_map=likelihood_map, behaviour=row['behaviour'],
                        dims=self.DIMENSIONS)

    def build_part(self, arr, name):
        pt = arr.to_numpy()
        if any(np.isnan(pt)):
            pt = np.array([MAGIC_NUMBER] * self.DIMENSIONS)
        return Part(pt, name, float(not all(pt == MAGIC_NUMBER)))

    @staticmethod
    def convert_to_list(index, skeleton, threshold=0.8):
        out = []
        for part in skeleton.body_parts:
            out.extend(skeleton[part].tolist() if skeleton[part] > threshold else ['', '', ''])
        return out
