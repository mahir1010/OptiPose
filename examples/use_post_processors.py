from OptiPose.data_store_interface import OptiPoseDataStore3D
from OptiPose.post_processor_interface import *

# Define Column names or body parts (order in csv file does not matter)
body_parts = ["snout", "rightEar", "leftEar", "headBase", "sp1", "sp2", "midpoint", "tailBase", "tailMid", "tailTip"]

data_file = OptiPoseDataStore3D(body_parts, 'recon_final.csv')

# Applying post processors

# The modules in post_processor_interface can be chained together.
# However, some of them might require statistics about the data file before running.
# You can check that by <class>.REQUIRES_STATS. If true, you need to run ClusterAnalysisProcessor before it.
# Some post processors are column based and can be parallelized independently.

# Threshold doesn't matter for OptiPose yet. It only supports binary values.
post_processors = [MedianDistanceFilterProcess(threshold=0.6, distance_threshold=300),
                   ClusterAnalysisProcess(threshold=0.6),
                   LinearInterpolationProcess("snout"),  # Not parallel here
                   LinearInterpolationProcess("rightEar"),  # Not parallel here
                   MovingAverageProcess("snout", window_size=3, threshold=0.6)
                   ]

for processor in post_processors:
    processor.process(data_file)
    data_file = processor.get_output()

data_file.save(path='recon_final_1.csv')
