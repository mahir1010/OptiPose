from OptiPose import OptiPoseConfig
from OptiPose.data_store_interface import FlattenedDataStore
from OptiPose.post_processor_interface import *

# Define Column names or body parts (order in csv file does not matter)
acinoset_config = OptiPoseConfig('./example_configs/AcinoSet.yml')

data_file = FlattenedDataStore(acinoset_config.body_parts, './AcinoSet_Files/20190227RomeoRun.csv')

# Applying post processors

# The modules in post_processor_interface can be chained together.
# However, some of them might require statistics about the data file before running.
# You can check that by <class>.REQUIRES_STATS. If true, you need to run ClusterAnalysisProcessor before it.
# Some post processors are column based and can be parallelized independently.

# Threshold doesn't matter for OptiPose yet. The data is either valid(>0) or invalid(<=0).
post_processors = [ClusterAnalysisProcess(threshold=0.6),
                   LinearInterpolationProcess("nose"),  # Not parallel here
                   LinearInterpolationProcess("l_eye"),  # Not parallel here
                   KalmanFilterProcess("l_eye", framerate=acinoset_config.framerate),
                   MovingAverageProcess("nose", window_size=3, threshold=0.6)
                   ]

for processor in post_processors:
    processor.process(data_file)
    data_file = processor.get_output()

data_file.save_file(f'{data_file.base_file_path}_processed.csv')
