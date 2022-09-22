from OptiPose.KalmanFilter import Tracker
from OptiPose.post_processor_interface.PostProcessorInterface import PostProcessorInterface


class KalmanFilterProcess(PostProcessorInterface):
    PROCESS_NAME = "Kalman Filtering"

    def __init__(self, target_column, framerate, skip=True, threshold=0.6):
        super(KalmanFilterProcess, self).__init__(target_column)
        self.threshold = threshold
        self.skip = skip
        self.dt = float(1 / framerate)
        self.tracker = None

    def process(self, data_store):
        self.data_store = data_store
        for index, point in self.data_store.part_iterator(self.target_column):
            self.progress = int(index / len(self.data_store) * 100)
            if self.skip:
                if point < self.threshold:
                    self.tracker = None
                    # self.data.append(None)
                else:
                    if self.tracker is None:
                        self.tracker = Tracker(point, self.dt)
                        # self.data.append(point.tolist())
                    else:
                        # self.data.append(self.tracker.update(point).tolist())
                        point[:3] = self.tracker.update(point).tolist()
                        self.data_store.set_marker(index, point)
            else:
                if point < self.threshold:
                    if self.tracker is not None:
                        p = self.tracker.get_next_pred()
                        p = self.tracker.update(p).tolist()
                        # self.data.append(p)
                        point[:3] = p
                        self.data_store.set_marker(index, point)
                    else:
                        self.data.append(None)
                else:
                    if self.tracker is None:
                        self.tracker = Tracker(point, self.dt)
                        # self.data.append(point.tolist())
                    else:
                        # self.data.append(self.tracker.update(point).tolist())
                        point[:3] = self.tracker.update(point).tolist()
                        self.data_store.set_marker(index, point)
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None
