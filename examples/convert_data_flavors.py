from OptiPose import convert_data_flavor
from OptiPose.data_store_interface import DannceDataStore, OptiPoseDataStore3D

dannce_bp = ["HeadF", "HeadB", "HeadL", "SpineF", "SpineM", "SpineL", "Offset1", "Offset2", "HipL", "HipR", "ShoulderL",
             "ShoulderR", "KneeR", "KneeL", "ShinL", "ShinR"]

datastore = DannceDataStore(dannce_bp, '/media/mahirp/Storage/Downloads/mocap-s1-d1.csv')

datastore_1 = OptiPoseDataStore3D(dannce_bp, '/media/mahirp/Storage/Downloads/mocap-s1-d1_reformat.csv')

# Output file should not exist
convert_data_flavor(datastore, datastore_1)
