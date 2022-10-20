from OptiPose import OptiPoseConfig
from OptiPose.data_store_interface import FlattenedDataStore, OptiPoseDataStore3D
from OptiPose.pipeline import convert_data_flavor

config = OptiPoseConfig('./example_configs/Rat7M.yml')

datastore = FlattenedDataStore(config.body_parts, '<path to source csv>')

datastore_1 = OptiPoseDataStore3D(config.body_parts, '<path to target csv>')

# Output file should not exist
convert_data_flavor(datastore, datastore_1)
