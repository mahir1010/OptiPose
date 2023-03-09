import json
import math

import numpy as np
import pandas as pd

from OptiPose import Part
from OptiPose.config import MAGIC_NUMBER, OptiPoseConfig, save_config


def rotate(p, rotation, scale=1.0, is_inv=False):
    assert type(scale) != bool
    if not is_inv:
        op = np.matmul(p, rotation) * np.array([1, 1, -1]) * scale
    else:
        op = np.matmul((p / scale) * np.array([1, 1, -1]), rotation)
    return op


def magnitude(vector):
    return np.linalg.norm(vector)


def compute_distance_matrix(skeleton):
    return np.array(
        [[magnitude(skeleton[p1] - skeleton[p2]) if skeleton[p1] > 0 and skeleton[p2] > 0 else -1 for p2 in
          skeleton.body_parts] for p1 in skeleton.body_parts])


def normalize_vector(vector):
    return np.divide(vector, magnitude(vector))


def get_spherical_coordinates(v1, is_degrees=True, is_shift=True):
    multiplier = 57.2958 if is_degrees else 1
    shift = (180 if is_degrees else math.pi) if is_shift else 0
    v1 = v1 / magnitude(v1)
    return np.array([math.atan2(v1[1], v1[0]),
                     math.atan2(math.sqrt(np.sum(np.square(v1[:2]))), v1[2])]) * multiplier + shift


def spherical_angle_difference(v1, v2, isAbs=True):
    diff = v1 - v2
    absDiff = np.abs(diff)
    output = np.minimum(absDiff, 360.0 - absDiff)
    if not isAbs:
        output *= np.sign(diff) * (
                (output == absDiff) * 2 - 1)  # Maps True/False to 1/-1 [T,F]*2-1 = [1,0]*2-1 = [2,0]*-1 = [1,-1]
    return output


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


def convert_to_numpy(input_data, dimensions=3):
    if type(input_data) == np.ndarray and (input_data.dtype == np.float32 or input_data.dtype == np.float32):
        return input_data
    if type(input_data) == list and len(input_data) == dimensions:
        return np.array(input_data)
    elif type(input_data) == str:
        split = ' '
        if ',' in input_data:
            split = ','
        input_data = np.array(input_data[1:-1].strip().split(split)).astype(np.float32)
    else:
        input_data = np.array([MAGIC_NUMBER] * dimensions, dtype=np.float32)
    return input_data


def build_input_data(csv_data, index, seq_len=60, size=10):
    PAD = np.array([[0, 0, 0]] * size)
    section = csv_data[index:min(index + seq_len, len(csv_data))].applymap(convert_to_list)
    data = list(map(convert_to_numpy, section.to_numpy()))
    while len(data) != seq_len:
        data.append(PAD)
    return np.array([data], dtype='float32')


def build_batch_input_data(csv_data, index, batch=12, seq_len=60):
    indices = []
    totalLen = len(csv_data)
    for i in range(index, min(index + seq_len * batch, totalLen), seq_len):
        indices.append(i)
        if i == index:
            inp = build_input_data(csv_data, i)
        else:
            inp = np.concatenate((inp, build_input_data(csv_data, i)), axis=0)
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
    data.save_file(f'{data.base_file_path}_offset.csv')


def convert_numpy_to_datastore(pickled_data: np.ndarray, header_names, flavor, output_path):
    assert pickled_data.ndim == 3 and pickled_data.shape[1] == len(header_names)
    from OptiPose.data_store_interface import initialize_datastore_reader
    datastore = initialize_datastore_reader(header_names, output_path, flavor)
    for index, row in enumerate(pickled_data):
        skeleton = datastore.get_skeleton(index)
        for i, header_name in enumerate(header_names):
            skeleton[header_name] = Part(row[i], header_name, 1.0)
        datastore.set_skeleton(index, skeleton)
    datastore.save_file()


def update_config_dlt_coeffs(config: OptiPoseConfig, dlt_coefficients_file, order):
    dlt_coefficients = pd.read_csv(dlt_coefficients_file, header=None)
    for index, camera in enumerate(order):
        dlt_coeff = dlt_coefficients[index].tolist()
        dlt_coeff.append(1)
        config.views[camera].dlt_coefficients = np.array(dlt_coeff)
    save_config(config.path, config.export_dict())

