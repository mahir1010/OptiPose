from OptiPose.KalmanFilter import Tracker
from OptiPose.post_processor_interface.PostProcessorInterface import PostProcessor


class KalmanFilterProcess(PostProcessor):
    PROCESS_NAME = "Kalman Filtering"

    def __init__(self, target_column, framerate, skip=True, threshold=0.6):
        super(KalmanFilterProcess, self).__init__(target_column)
        self.threshold = threshold
        self.skip = skip
        self.dt = float(1 / framerate)

    def process(self, data_store):
        self.data_store = data_store
        self.data_ready = False
        self.progress = 0
        tracker = None
        for index, point in self.data_store.part_iterator(self.target_column):
            self.progress = int(index / len(self.data_store) * 100)
            if self.skip:
                if point < self.threshold:
                    # new_tracker = Tracker(point, self.dt)
                    # new_tracker.tracker.R = tracker.tracker.R.copy()
                    # new_tracker.tracker.Q = tracker.tracker.Q.copy()
                    # new_tracker.tracker.P = tracker.tracker.P.copy()
                    del tracker
                    # tracker = new_tracker
                    tracker = None
                else:
                    if tracker is None:
                        tracker = Tracker(point, self.dt)
                        # self.data.append(point.tolist())
                    else:
                        # self.data.append(tracker.update(point).tolist())
                        point[:3] = tracker.update(point).tolist()
                        self.data_store.set_part(index, point)
            else:
                if point < self.threshold:
                    if tracker is not None:
                        p = tracker.get_next_pred()
                        p = tracker.update(p).tolist()
                        # self.data.append(p)
                        point[:3] = p
                        self.data_store.set_part(index, point)
                    else:
                        self.data.append(None)
                else:
                    if tracker is None:
                        tracker = Tracker(point, self.dt)
                        # self.data.append(point.tolist())
                    else:
                        # self.data.append(tracker.update(point).tolist())
                        point[:3] = tracker.update(point).tolist()
                        self.data_store.set_part(index, point)
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None
