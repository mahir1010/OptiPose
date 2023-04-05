from cvkit.pose_estimation.data_readers import FlattenedDataStore, CVKitDataStore3D
from cvkit.pose_estimation.data_readers import convert_data_flavor

from cvkit.pose_estimation.config import PoseEstimationConfig

config = PoseEstimationConfig('./example_configs/Rat7M.yml')

datastore = FlattenedDataStore(config.body_parts, '<path to source csv>')

datastore_1 = CVKitDataStore3D(config.body_parts, '<path to target csv>')

# Output file should not exist
convert_data_flavor(datastore, datastore_1)
