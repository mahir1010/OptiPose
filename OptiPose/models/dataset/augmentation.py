import math
from random import randint, sample, uniform

import numpy as np
import scipy

from OptiPose import MAGIC_NUMBER


def aug_alternate_missing(dataset, BLANK):
    if randint(1, 2) == 1:
        missing_index = list(range(1, len(dataset) + 1, 2))
    else:
        missing_index = list(range(0, len(dataset) + 1, 2))
    for idx in range(len(dataset)):
        if idx in missing_index:
            dataset[idx] = [BLANK.copy() for i in range(len(dataset[0]))]
    mask = [len(missing_index) for kp in range(len(dataset[0]))]
    return dataset, mask


def aug_sentinals_missing(dataset, BLANK):
    n_poses = len(dataset)
    max_missing = max(2,n_poses // 2)
    min_missing = max(1, max_missing // 3)
    direction = randint(1, 2) == 1
    if direction:
        missing_index = list(range(0, randint(min_missing, max_missing)))
    else:
        missing_index = list(range(n_poses, n_poses - randint(min_missing, max_missing)))
    for idx in range(len(dataset)):
        if idx in missing_index:
            dataset[idx] = [BLANK.copy() for i in range(len(dataset[0]))]
    mask = [len(missing_index) for kp in range(len(dataset[0]))]
    return dataset, mask


def aug_clusters_missing(dataset, BLANK):
    n_poses = len(dataset)
    n_kps = len(dataset[0])
    cluster_size = randint(2, min(10, n_poses // 3))
    cluster_start = sample(list(range(0, n_poses, cluster_size)), 1)
    missing_index = []
    mask = [0] * n_kps
    for c in cluster_start:
        missing_index.extend([m for m in range(c, min(c + cluster_size, len(dataset)))])
    for idx in range(len(dataset)):
        if idx in missing_index:
            dataset[idx] = [BLANK.copy() for i in range(len(dataset[0]))]
        else:
            missing_kps = sample(list(range(n_kps)), randint(0, 3))
            dataset[idx] = [dataset[idx][n] if n not in missing_kps else BLANK.copy() for n in range(n_kps)]
            mask = [mask[i] + 1 if i in missing_kps else mask[i] for i in range(n_kps)]
    mask = [mask[kp] + len(missing_index) for kp in range(n_kps)]
    return dataset, mask


def aug_auto_encoder(dataset, BLANK):
    return dataset, [0 for kp in range(len(dataset[0]))]


def aug_kp_cluster_missing(dataset, BLANK):
    n_kps = len(dataset[0])
    n_poses = len(dataset)
    complete = randint(1, 2) == 1
    missing_kps = sample(list(range(n_kps)), randint(1, 3))
    cluster_size = randint(n_poses // 3, n_poses - 5)
    mask = [0 for kp in range(n_kps)]
    if complete:
        missing_index = list(range(n_poses))
        for idx in missing_index:
            dataset[idx] = [dataset[idx][n] if n not in missing_kps else BLANK.copy() for n in range(n_kps)]
            for n in missing_kps:
                mask[n] += 1
    else:
        cluster_start = sample(list(range(0, n_poses, cluster_size)), min(1, math.ceil(n_poses / cluster_size)))
        missing_index = []
        missing_kps = sample(list(range(n_kps)), randint(1, 3))
        for c in cluster_start:
            missing_index.extend([m for m in range(c, min(c + cluster_size, len(dataset)))])
        for idx in range(n_poses):
            if idx in missing_index:
                dataset[idx] = [dataset[idx][n] if n not in missing_kps else BLANK.copy() for n in range(n_kps)]
                for n in missing_kps:
                    mask[n] += 1
            else:
                random_missing_kps = sample(list(range(n_kps)), randint(0, 3))
                dataset[idx] = [dataset[idx][n] if n not in random_missing_kps else BLANK.copy() for n in range(n_kps)]
                mask = [mask[i] + 1 if i in random_missing_kps else mask[i] for i in range(n_kps)]
    return dataset, mask


def aug_kp_missing(dataset, BLANK):
    n_kps = len(dataset[0])
    n_poses = len(dataset)
    missing_index = sample(list(range(n_poses)), randint(n_poses - 10, n_poses - 5))
    mask = [0 for kp in range(n_kps)]
    for idx in missing_index:
        grouped = randint(1, 10) == 1
        if grouped:
            missing_kps = sample(list(range(n_kps)), 1)
            temp = [list(range(m, min(m + 3, n_kps))) for m in missing_kps]
            missing_kps = [item for cluster_kp in temp for item in cluster_kp]
        else:
            missing_kps = sample(list(range(n_kps)), randint(1, n_kps // 3))
        dataset[idx] = [dataset[idx][n] if n not in missing_kps else BLANK.copy() for n in range(n_kps)]
        for n in missing_kps:
            mask[n] += 1
    return dataset, mask


def random_rigid_transformation(inputs, labels, rotation=0, min_x=-50, max_x=1100, min_y=-50, max_y=1100,
                                static_scale=1.0):
    rotation_axis = [0, 0, rotation]
    rotation_mat = scipy.spatial.transform.Rotation.from_euler("XYZ", rotation_axis, degrees=True).as_matrix()
    x = [99999, -99999]
    y = [99999, -99999]
    for row in range(len(inputs)):
        labels[row] = [static_scale * np.matmul(part, rotation_mat) for part in labels[row]]
    for row in range(len(labels)):
        for col, point in enumerate(labels[row]):
            x[0] = min(x[0], point[0])
            x[1] = max(x[1], point[0])
            y[0] = min(y[0], point[1])
            y[1] = max(y[1], point[1])
    translation_vector = np.array(
        [-x[0] + randint(min_x, int(max_x - x[1] + x[0])), -y[0] + randint(min_y, int(max_y - y[1] + y[0])),
         randint(0, 20)])
    for row in range(len(labels)):
        for col in range(len(labels[row])):
            labels[row][col] = (labels[row][col] + translation_vector).tolist()
            if not all([v == MAGIC_NUMBER for v in inputs[row][col]]):
                inputs[row][col] = labels[row][col]
    return inputs, labels


def add_random_noise(inputs, noise_val=15, is_auto_encoder=False):
    rand_max = 10 if not is_auto_encoder else 5
    noise_array = np.array(
        [[uniform(-noise_val, noise_val), uniform(-noise_val, noise_val), uniform(-noise_val, noise_val)] for i in
         range(len(inputs[0]))])
    is_cluster = randint(1, 10) == 1 if not is_auto_encoder else randint(1, 5) == 1
    prob_array = [randint(1, rand_max) == 1 for i in range(len(inputs[0]))]
    for row in range(len(inputs)):
        for col in range(len(inputs[row])):
            if not all([v == MAGIC_NUMBER for v in inputs[row][col]]):
                if is_cluster and prob_array[col]:
                    inputs[row][col] = (inputs[row][col] + noise_array[col]).tolist()
                elif randint(1, rand_max) == 1:
                    inputs[row][col] = (inputs[row][col] + np.array(
                        [uniform(-noise_val, noise_val), uniform(-noise_val, noise_val),
                         uniform(-noise_val, noise_val)])).tolist()
                elif randint(1, 2) == 1:
                    inputs[row][col] = (
                            inputs[row][col] + np.array([uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)])).tolist()
    return inputs
