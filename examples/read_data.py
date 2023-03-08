from OptiPose.data_store_interface import initialize_datastore_reader
from OptiPose.video_reader_interface import initialize_video_reader
from OptiPose import OptiPoseConfig
import matplotlib.pyplot as plt
import cv2

config = OptiPoseConfig('./example_configs/Rodent3D.yml')

datastore = initialize_datastore_reader(config.body_parts,'<OptiPose3D_csv_path>','OptiPose3D')

for index,skeleton in datastore.row_iterator():
    #You can use bodypart name as key to retrieve that keypoint
    print(skeleton['snout'],skeleton['rightEar'])
    break

#Load Video
video_reader = initialize_video_reader('<video_path>',60,'opencv')
#Start reading
video_reader.start()
plt.ion()
while True:
    frame = video_reader.next_frame()
    frame_number = video_reader.get_current_index()
    if frame is None:
        break
    print(frame_number)
    plt.imshow(frame)
    plt.pause(1/60)