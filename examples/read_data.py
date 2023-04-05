import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from cvkit.pose_estimation.data_readers import initialize_datastore_reader
from cvkit.video_readers import initialize_video_reader

from cvkit.pose_estimation.config import PoseEstimationConfig

config = PoseEstimationConfig('./example_configs/Rodent3D.yml')

datastore = initialize_datastore_reader(config.body_parts, '<OptiPose3D_csv_path>', 'OptiPose3D')

for index, skeleton in datastore.row_iterator():
    # You can use bodypart name as key to retrieve that keypoint
    print(skeleton['snout'], skeleton['rightEar'])
    break

# Load Video
video_reader = initialize_video_reader('<video file path>', 60, 'opencv')
# Start reading
video_reader.start()
plt.ion()
# Load Depth
data_file = h5py.File('<corresponding depth hdf5 path>', 'r')
dataset = data_file['depth']  # Select depth group
while True:
    frame = video_reader.next_rame()  # Read Frame
    frame_number = video_reader.get_current_index()  # Get Frame number
    if frame is None:
        break
    depth_image = dataset[frame_number]  # Read corresponding depth frame
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1),
                                       cv2.COLORMAP_JET)  # Map depth values to valid RGB range
    images = np.hstack((frame, depth_colormap))  # Stack frames side by side
    print(frame_number)
    plt.imshow(images)  # Show frames
    plt.pause(1 / 60)
