import numpy as np

from OptiPose.post_processor_interface.PostProcessorInterface import PostProcessorInterface


class MovingAverageProcess(PostProcessorInterface):
    PROCESS_NAME = "Moving Average"

    def __init__(self, target_column, window_size, threshold=0.6):
        super(MovingAverageProcess, self).__init__(target_column)
        self.threshold = threshold
        self.window_size = window_size

    def process(self, data_store):
        self.data_store = data_store
        average_window = []
        self.data_ready = False
        self.progress = 0
        for index, point in self.data_store.part_iterator(self.target_column):
            self.progress = int(index / len(self.data_store) * 100)
            if point < self.threshold:
                average_window.clear()
            else:
                average_window.append(point)
                if len(average_window) > self.window_size:
                    average_window.pop(0)
                point[:] = np.mean(average_window, axis=0)
                self.data_store.set_marker(index, point)
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None
