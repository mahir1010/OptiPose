from abc import ABC, abstractmethod

from OptiPose.data_store_interface import DataStoreInterface


class PostProcessorInterface(ABC):
    REQUIRES_STATS = False
    PROCESS_NAME = "Abstract"
    PRINT = False

    def __init__(self, target_column=None):
        self.target_column = target_column
        self.progress = 0
        self.data_store = None
        self.data_ready = False

    @abstractmethod
    def process(self, data_store: DataStoreInterface):
        pass

    def get_progress(self):
        return self.progress

    @abstractmethod
    def get_output(self):
        pass
