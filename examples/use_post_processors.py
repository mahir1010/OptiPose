from cvkit.pose_estimation.data_readers import FlattenedDataStore
from cvkit.pose_estimation.processors.util import *
from cvkit.pose_estimation.processors.filter import *
from cvkit.pose_estimation.processors.generative import *

from cvkit.pose_estimation.config import PoseEstimationConfig

# Define Column names or body parts (order in csv file does not matter)
acinoset_config = PoseEstimationConfig('./example_configs/AcinoSet.yml')

data_file = FlattenedDataStore(acinoset_config.body_parts, './AcinoSet_Files/20190227RomeoRun.csv')

# Applying post processors

# The modules in post_processor_interface can be chained together.
# However, some of them might require statistics about the data file before running.
# You can check that by <class>.REQUIRES_STATS. If true, you need to run ClusterAnalysisProcessor before it.
# Some post processors are column based and can be parallelized independently.

# Threshold doesn't matter for OptiPose yet. The data is either valid(>0) or invalid(<=0).
post_processors = [ClusterAnalysis(threshold=0.6),
                   LinearInterpolationFilter("nose"),  # Not parallel here
                   LinearInterpolationFilter("l_eye"),  # Not parallel here
                   KalmanFilter("l_eye", framerate=acinoset_config.framerate),
                   MovingAverageFilter("nose", window_size=3, threshold=0.6)
                   ]

for processor in post_processors:
    processor.process(data_file)
    data_file = processor.get_output()

data_file.save_file(f'{data_file.base_file_path}_processed.csv')
