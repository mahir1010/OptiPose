from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

from OptiPose.skeleton import Skeleton
from OptiPose.utils import *


class Tracker:
    def __init__(self, data, dt):
        self.dt = dt
        self.tracker = self.getKalmanFilter(data)

    def getKalmanFilter(self, data):
        kalman = KalmanFilter(len(data) * 2, len(data))
        kalman.x = np.hstack((data, [0.0, 0.0, 0.0])).astype(np.float)
        kalman.F = np.array(
            [[1, 0, 0, self.dt, 0, 0], [0, 1, 0, 0, self.dt, 0], [0, 0, 1, 0, 0, self.dt], [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]])
        kalman.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        kalman.P *= 1000
        kalman.R = 0.00001
        kalman.Q = Q_discrete_white_noise(2, dt=self.dt, var=0.5, block_size=3)
        kalman.B = 0
        return kalman

    def get_next_pred(self):
        return self.tracker.H @ self.tracker.get_prediction()[0]

    def update(self, data, likelihood=1.0, threshold=0.9):
        self.tracker.predict()
        if likelihood < threshold:
            self.tracker.update(None)
        else:
            self.tracker.update(np.array(data))
        return self.tracker.x[:3]


class SkeletonTracker:
    def __init__(self, skeleton: Skeleton, dt):
        self.parts = {}
        self.dt = dt
        for part in skeleton.parts.keys():
            self.parts[part] = Tracker(skeleton[part])

    def get_next_pred(self, part):
        return self.parts[part].get_next_pred()

    def update(self, skeleton: Skeleton, threshold=.80):
        for part in self.parts:
            skeleton[part] = self.parts[part].update(skeleton[part], skeleton.partsLikelihood[part], threshold)
        return skeleton
