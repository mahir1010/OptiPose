from OptiPose.data_store_interface import DataStoreInterface
from OptiPose.post_processor_interface import PostProcessor


class FileLoadProcess(PostProcessor):
    PROCESS_NAME = "Load File"

    def process(self, data_store: DataStoreInterface):
        self.progress = 100
        pass

    def get_output(self):
        return self.data_store

    def __init__(self, data_store):
        super(FileLoadProcess, self).__init__(None)
        self.data_store = data_store
