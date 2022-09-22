import json
import math
from math import sqrt

import numpy as np

from OptiPose.config import MAGIC_NUMBER


def rotate(p, rotation, scale=1.0, is_inv=False):
    assert type(scale) != bool
    if not is_inv:
        op = np.matmul(p, rotation) * np.array([1, 1, -1]) * scale
    else:
        op = np.matmul((p / scale) * np.array([1, 1, -1]), rotation)
    return op


def magnitude(vector):
    return sqrt(np.sum(np.square(vector))) + 1e-5


def evaluateFunction(fn, params):
    if fn.strip().lower() == "avg":
        return np.mean(params, axis=0)


def alphaBetaFilter(measurement, prevState, dt, velocity=0, acceleration=0, a=0.7, b=0.85, g=0.8):
    estimate = prevState + velocity * dt + 0.5 * (dt ** 2) * acceleration
    velocity = velocity + acceleration * dt
    residual = measurement - estimate
    estimate = estimate + residual * a
    velocity = residual * (b / dt) + velocity
    acceleration = acceleration + g * residual
    return estimate, velocity, acceleration


def vectorYawPitch(v1, isDegrees=True):
    multiplier = 57.2958 if isDegrees else 1
    yaw = multiplier * math.atan2(v1[1], v1[0])
    # yaw = yaw + (multiplier *2 *math.pi) if yaw<0 else yaw
    return np.array([yaw, 0])


def getCosineAngle(v1, v2, isDegrees=True):
    multiplier = 57.2958 if isDegrees else 1
    v1 = v1[:2]
    v2 = v2[:2]
    return math.acos(np.dot(v1, v2) / (magnitude(v1) * magnitude(v2))) * multiplier


def angleBVectors(v1, v2, isDegrees=True):
    multiplier = 57.2958 if isDegrees else 1
    # Yaw Calculations
    y1 = multiplier * math.atan2(v1[1], v1[0])
    y2 = multiplier * math.atan2(v2[1], v2[0])
    yaw = y2 - y1
    # pitch = multiplier * math.acos((v2[2]-v1[2])/magnitude(v2))
    return np.array([yaw, 0])


def getVectorMatrix(parts, rodent):
    for i, p in enumerate(parts[:-1]):
        for j, p2 in enumerate(parts[i + 1:]):
            print(p, '-', p2, ':', angleBVectors(rodent[p2], rodent[p]), end=' ')
        print(' ')
    print('-------------------')


def convert_to_list(inp):
    if type(inp) != str and type(inp) != list and math.isnan(inp):
        t = np.array([MAGIC_NUMBER, MAGIC_NUMBER, MAGIC_NUMBER], dtype=np.float32)
    else:
        if type(inp) != str:
            t = np.array(inp, dtype=np.float32)
        else:
            if ',' not in inp:
                inp = inp.replace(' ', ',')
            t = np.array(json.loads(inp)).astype(np.float32)
    return t


def convert_to_numpy(input_data):
    if type(input_data) == np.ndarray and (input_data.dtype == np.float32 or input_data.dtype == np.float32):
        return input_data
    if type(input_data) == np.ndarray:
        input_data = np.array(list(map(convert_to_numpy, input_data)), dtype=np.float32)
    elif type(input_data) == str:
        split = ' '
        if ',' in input_data:
            split = ','
        input_data = np.array(input_data[1:-1].strip().split(split)).astype(np.float32)
    else:
        input_data = np.array([MAGIC_NUMBER, MAGIC_NUMBER, MAGIC_NUMBER], dtype=np.float32)
    return input_data


def buildInputData(csv_data, index, seq_len=60, size=10):
    PAD = np.array([[0, 0, 0]] * size)
    section = csv_data[index:min(index + seq_len, len(csv_data))].applymap(convert_to_list)
    data = list(map(convert_to_numpy, section.to_numpy()))
    while len(data) != seq_len:
        data.append(PAD)
    return np.array([data], dtype='float32')


def buildBatchInputData(csv_data, index, batch=12, seq_len=60, size=10):
    PAD = np.array([[0, 0, 0]] * size)
    indices = []
    totalLen = len(csv_data)
    for i in range(index, min(index + seq_len * batch, totalLen), seq_len):
        indices.append(i)
        if i == index:
            inp = buildInputData(csv_data, i)
        else:
            inp = np.concatenate((inp, buildInputData(csv_data, i)), axis=0)
    return indices, inp


def offset_deeplabcut_csv(path, parts, offset_x, offset_y):
    from OptiPose.data_store_interface import DeeplabcutDataStore
    data = DeeplabcutDataStore(parts, path)
    for index, sk in data.row_iterator():
        for part in parts:
            sk[part][0] += offset_x
            sk[part][1] += offset_y
        data.set_skeleton(index, sk, force_insert=True)
        print(f'\r{index}/{len(data)}', end='')
    data.save_file('/media/mahirp/Storage/RodentVideos/TestVideos/220510/Green_3.csv_data')


if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(open('/media/mahirp/Storage/RodentVideos/TestVideos/220510/test.yaml', 'r'))
    # offset_deeplabcut_csv(
    #     '/media/mahirp/Storage/RodentVideos/TestVideos/220510/GreenDLC_effnet_b0_FLIRDLC_GREENAug6shuffle1_70000.csv_data',
    #     config['body_parts'], 0, 110)
    # video_reader = initialize_video_reader('/media/mahirp/Storage/RodentVideos/051822/Pink.mp4',60,'opencv')
    # compute_invalid_frame_indices(video_reader,10)
    # video_reader = initialize_video_reader('/media/mahirp/Storage/RodentVideos/051822/Blue.mp4', 60, 'opencv')
    # compute_invalid_frame_indices(video_reader, 10)
    # data_reader = initialize_datastore_reader(config['body_parts'],'/media/mahirp/Storage/RodentVideos/051822/YellowDLC_resnet50_RGBModelMay27shuffle1_350000.csv','deeplabcut')
    # invalid_indices = pd.read_csv('/media/mahirp/Storage/RodentVideos/051822/Yellow.csv',dtype=bool)
    # filter_invalid_frames(data_reader,invalid_indices)
    # csv_maps = {}
    # for cam in config['views']:
    #     view_data = config['views'][cam]
    #     csv_maps[cam] = DeeplabcutDataStore(config['body_parts'], view_data['annotation_file'])
    #
    # generateEasyWandData(config, csv_maps)
