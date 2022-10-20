import os

from OptiPose import OptiPoseConfig
from OptiPose.data_store_interface import FlattenedDataStore
from OptiPose.models.dataset.generate_dataset import generate_dataset

# Download dataset files from https://drive.google.com/drive/folders/1Gg08RiEa-As_lDR5xyBvIjLbp4FEAUt9?usp=sharing

is_test = False

DF3D_config = OptiPoseConfig('./example_configs/DF3D.yml')
base_folder = "./DF3D_Files"
if not is_test:
    csv_files = ['df3d_001.csv', 'df3d_003.csv', 'df3d_006.csv', 'df3d_007.csv', 'df3d_008.csv', 'df3d_009.csv',
                 'df3d_010.csv', 'df3d_011.csv']
else:
    csv_files = ['df3d_004.csv', 'df3d_005.csv']
data_stores = [FlattenedDataStore(DF3D_config.body_parts, os.path.join(base_folder, path)) for path in csv_files]
generate_dataset(os.path.join(base_folder, 'Dataset'), "DF3D", is_test, data_stores, 20000, min_seq_length=20,
                 max_seq_length=30, min_x=-700,
                 max_x=700, min_y=-700, max_y=700, prefix="IJCV", suffix="dataset_t_attention", random_noise=50)
