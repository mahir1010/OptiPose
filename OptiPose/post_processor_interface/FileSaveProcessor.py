from OptiPose.data_store_interface import DataStoreInterface
from OptiPose.post_processor_interface import PostProcessorInterface


class FileSaveProcess(PostProcessorInterface):
    PROCESS_NAME = "Save File"

    def process(self, data_store: DataStoreInterface):
        self.data_store = data_store
        data_store.save_file(self.path)
        self.progress = 100

    def get_output(self):
        return self.data_store

    def __init__(self, path):
        super(FileSaveProcess, self).__init__(None)
        self.path = path
