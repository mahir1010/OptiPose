import numpy as np

from OptiPose import Part
from OptiPose.data_store_interface import FlattenedDataStore
from OptiPose.post_processor_interface.PostProcessorInterface import PostProcessor


class HeadingAnalysisProcess(PostProcessor):
    PROCESS_NAME = "Heading Analysis"

    def __init__(self, head_directon: list, body_direction: list, movement_direction: list, motion_smooth_window=10):
        super(HeadingAnalysisProcess, self).__init__()
        self.head_direction = head_directon
        self.body_direction = body_direction
        self.movement_direction = movement_direction
        self.motion_smooth_window = motion_smooth_window

    def process(self, data_store):
        self.data_store = FlattenedDataStore(['hd', 'bd', 'md'], None)
        self.data_ready = False
        self.progress = 0
        previous_point = None
        moving_avg_motion = []
        for index, skeleton in data_store.row_iterator():
            self.progress = int(index / len(data_store) * 100)
            if self.PRINT and self.progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self.progress}% complete', end='')
            head_direction = Part(skeleton[self.head_direction[1]] - skeleton[self.head_direction[0]], 'hd', 1.0)
            body_direction = Part(skeleton[self.body_direction[1]] - skeleton[self.body_direction[0]], 'bd', 1.0)
            position = np.mean([skeleton[part] for part in self.movement_direction], axis=0)
            if previous_point is None:
                movement_direction = Part([0, 0, 0], 'md', 1.0)
            else:
                moving_avg_motion.append((position - previous_point).tolist())
                if len(moving_avg_motion) > self.motion_smooth_window:
                    moving_avg_motion.pop(0)
                movement_direction = Part(np.mean(moving_avg_motion, axis=0), 'md', 1.0)
            self.data_store.set_part(index, head_direction)
            self.data_store.set_part(index, body_direction)
            self.data_store.set_part(index, movement_direction)
            previous_point = position
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None
