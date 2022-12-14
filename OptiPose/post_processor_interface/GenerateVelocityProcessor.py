import numpy as np

from OptiPose import MAGIC_NUMBER, Part
from OptiPose.post_processor_interface.PostProcessorInterface import PostProcessorInterface


class GenerateVelocityProcess(PostProcessorInterface):
    PROCESS_NAME = "Generate Velocity"

    def __init__(self, target_column, frame_rate, velocity_threshold, threshold=0.6):
        super(GenerateVelocityProcess, self).__init__(target_column)
        self.dt = 1 / frame_rate
        self.threshold = threshold
        self.velocity_threshold = velocity_threshold

    def process(self, data_store, empty_datastore):
        self.data_store = data_store
        self.data_ready = False
        self.progress = 0
        previous_point = None
        previous_index = -1
        for index, point in self.data_store.part_iterator(self.target_column):
            self.progress = int(index / len(self.data_store) * 100)
            if self.PRINT and self.progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self.progress}% complete', end='')
            velocity = [MAGIC_NUMBER, MAGIC_NUMBER, MAGIC_NUMBER]
            flag = False
            if point > self.threshold:
                if previous_point is not None:
                    velocity = np.subtract(point, previous_point) / ((index - previous_index) * self.dt)
                else:
                    velocity = [0, 0, 0]
                if max(np.abs(velocity).tolist()) <= self.velocity_threshold:
                    flag = True
                    previous_point = point.copy()
                    previous_index = index
                else:
                    velocity = [MAGIC_NUMBER, MAGIC_NUMBER, MAGIC_NUMBER]
            empty_datastore.set_marker(index, Part(velocity, self.target_column, float(flag)))
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None
