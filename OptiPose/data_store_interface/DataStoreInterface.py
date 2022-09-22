from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from OptiPose import MAGIC_NUMBER
from OptiPose.skeleton import Skeleton, Part


class DataStoreInterface(ABC):
    FLAVOR = "Abstract"
    DIMENSIONS = 3
    SEP = ','

    def __init__(self, body_parts, path):
        """
        Interface for data reader. This class can be implemented to integrate data files from other toolkits
        Args:
            body_parts: list of column names
            path: path to data file
        """
        self.body_parts = body_parts
        self.data = None
        self.path = path
        self.stats = DataStoreStats(body_parts)

    def get_skeleton(self, index) -> Skeleton:
        if index in self.data.index:
            return self.build_skeleton(self.data.loc[index])
        else:
            return self.build_empty_skeleton()

    def set_skeleton(self, index, skeleton: Skeleton, force_insert=False) -> None:
        insert = True
        if not force_insert and index not in self.data.index:
            insert = False
            for part in self.body_parts:
                if skeleton[part] > 0:
                    insert = True
                    break
        if insert or force_insert:
            for part in self.body_parts:
                self.set_marker(index, skeleton[part])
            self.set_behaviour(index, skeleton.behaviour)

    def get_numpy(self, index):
        s = self.get_skeleton(index)
        arr = [np.array(s[part]) for part in self.body_parts]
        return np.array(arr)

    def delete_skeleton(self, index):
        if index in self.data.index:
            for part in self.body_parts:
                self.delete_marker(index, part, True)

    @abstractmethod
    def set_behaviour(self, index, behaviour: str) -> None:
        pass

    @abstractmethod
    def get_behaviour(self, index) -> str:
        pass

    @abstractmethod
    def get_keypoint_slice(self, slice_indices: list, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def set_keypoint_slice(self, slice_indices: list, name: str, data: np.ndarray) -> None:
        pass

    def row_iterator(self):
        for index, row in self.data.iterrows():
            yield index, self.build_skeleton(row)

    def part_iterator(self, part):
        for index, row in self.data[part].items():
            yield index, self.build_part(row, part)

    @abstractmethod
    def get_marker(self, index, name) -> Part:
        pass

    @abstractmethod
    def set_marker(self, index, part: Part) -> None:
        pass

    @abstractmethod
    def delete_marker(self, index, name, force_remove=False):
        pass

    @abstractmethod
    def build_skeleton(self, row) -> Skeleton:
        pass

    @abstractmethod
    def build_part(self, arr, name) -> Part:
        pass

    def save_file(self, path: str = None) -> None:
        if path is None:
            path = self.path
        self.data.sort_index(inplace=True)
        self.data.to_csv(path)

    def set_stats(self, stats):
        if stats.register(self.compute_data_hash()):
            del self.stats
            self.stats = stats

    def build_empty_skeleton(self):
        part_map = {}
        likelihood_map = {}
        for name in self.body_parts:
            part_map[name] = [MAGIC_NUMBER] * self.DIMENSIONS
            likelihood_map[name] = 0.0
        return Skeleton(self.body_parts, part_map=part_map, likelihood_map=likelihood_map, behaviour='',
                        dims=self.DIMENSIONS)

    def __len__(self):
        return len(self.data)

    def compute_data_hash(self):
        return int(pd.util.hash_pandas_object(self.data).sum())

    def verify_stats(self):
        return self.compute_data_hash() == self.stats.data_frame_hash

    @staticmethod
    @abstractmethod
    def convert_to_list(index, skeleton, threshold=0.8):
        pass

    def get_header_rows(self):
        return [self.data.columns.tolist()]


class DataStoreStats:

    def __init__(self, body_parts):
        self.data_frame_hash = 0
        self.body_parts = body_parts
        self.na_data_points = {}
        self.accurate_data_points = []
        self._na_current_cluster = {}
        for column in body_parts:
            self.na_data_points[column] = []
            self._na_current_cluster[column] = {'begin': -2, 'end': -2}
        self._accurate_cluster = {'begin': -2, 'end': -2}
        self.registered = False

    def update_cluster_info(self, index, part, accurate=False):
        cluster = self._na_current_cluster[part] if not accurate else self._accurate_cluster
        data_point = self.na_data_points[part] if not accurate else self.accurate_data_points
        if cluster['end'] + 1 == index:
            cluster['end'] = index
        else:
            if cluster['begin'] != -2:
                data_point.append(cluster.copy())
            cluster['begin'] = cluster['end'] = index

    def register(self, data_frame_hash):
        if not self.registered:
            for col in self._na_current_cluster.keys():
                if self._na_current_cluster[col]['begin'] != -2:
                    self.na_data_points[col].append(self._na_current_cluster[col].copy())
            if self._accurate_cluster['begin'] != -2:
                self.accurate_data_points.append(self._accurate_cluster.copy())
            del self._na_current_cluster, self._accurate_cluster
            self.data_frame_hash = data_frame_hash
            self.registered = True
            return True
        return False

    def iter_na_clusters(self, part):
        for candidate in self.na_data_points[part]:
            yield candidate

    def iter_accurate_clusters(self):
        for accurate in self.accurate_data_points:
            yield accurate
