import os
from abc import ABC, abstractmethod

import numpy as np


class BaseVideoReaderInterface(ABC):
    FLAVOR = "Abstract"

    def __init__(self, video_path, fps):
        self.video_path = video_path
        self.base_file_path = os.path.splitext(self.video_path)[0]
        self.fps = fps
        self.total_frames = -1

    @abstractmethod
    def seek_pos(self, index: int) -> None:
        pass

    @abstractmethod
    def next_frame(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_current_index(self) -> int:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def pause(self) -> None:
        pass

    @abstractmethod
    def get_number_of_frames(self) -> int:
        pass

    @abstractmethod
    def random_access_image(self, position):
        pass

    def __len__(self):
        return self.get_number_of_frames()
