import csv
import os
from copy import deepcopy

import matplotlib.pyplot as plt
from cvkit.pose_estimation.processors.util import ClusterAnalysis

from OptiPose.model.dataset.augmentation import *
from OptiPose.model.utils import build_batch

TEST_FILE_SIZE = 6500


def generate_dataset(root_path, dataset_name, is_test, data_stores: list, total_samples, max_seq_length=60,
                     min_seq_length=2, prefix="", suffix="", min_x=-100, max_x=1100, min_y=-100, max_y=1100, scale=1.0,
                     random_noise=10, min_standard_deviation=None, max_skip=0, truncate=False):
    assert 2 < min_seq_length <= max_seq_length
    file_name = f"{prefix}_{dataset_name}_{suffix}_{max_seq_length}_{total_samples}{'_test' if is_test else ''}.csv"
    # samples_per_file = total_samples // len(data_stores) if not is_test else (TEST_FILE_SIZE // len(data_stores))
    total_samples = total_samples if not is_test else TEST_FILE_SIZE
    writer = csv.writer(open(os.path.join(root_path, file_name), 'w'), delimiter='|')
    writer.writerow(['input', 'label'])
    BLANK = [MAGIC_NUMBER] * 3
    PAD_VAL = [0, 0, 0]
    PAD = [PAD_VAL.copy() for p in data_stores[0].body_parts]

    body_parts = data_stores[0].body_parts.copy()

    avg_length_list = min_seq_length
    analysis_processor = ClusterAnalysis()
    analysis_processor.PRINT = True
    batch_rows = []
    avg_mask = np.array([0] * len(body_parts))
    total_count = 0
    histogram_list = []
    cluster_counts = []
    for ds_index, data_store in enumerate(data_stores):
        if not data_store.verify_stats():
            print(f'\nAnalyzing File for accurate data points: {ds_index}/{len(data_stores)}')
            analysis_processor.process(data_store)
        c_count, histogram, _ = data_store.stats.get_accurate_cluster_info()
        cluster_counts.append(c_count)
        histogram_list.append(histogram)
    for histogram in histogram_list[1:]:
        for key in histogram_list[0]:
            histogram_list[0][key] += histogram[key]
    print(f"\nGenerating datasets from {np.sum(cluster_counts).round(0)} clusters: Histogram {histogram_list[0]}")
    for ds_index, data_store in enumerate(data_stores):
        count = 0
        # samples_per_file = total_samples // (np.sum(cluster_counts) / cluster_counts[ds_index]) + 1
        samples_per_file = total_samples // len(data_stores) + 1
        print(f"\nExtracting {samples_per_file} from file: {data_store.path}")
        flag = True
        while flag:
            iteration_count = 0
            for dp in data_store.stats.iter_accurate_clusters():
                skip = (sample(list(range(max_skip)), 1)[0] if max_skip != 0 else 0) + 1
                while skip > 1:
                    if dp['end'] - dp['begin'] < min_seq_length * skip:
                        skip -= 1
                    else:
                        break

                if count >= samples_per_file:
                    flag = False
                    break
                if count % 200 == 0:
                    print(f'\r {total_count}/{total_samples}', end='')
                if dp['end'] - dp['begin'] > min_seq_length * skip:
                    begin = randint(dp['begin'], dp['end'] - min_seq_length * skip)
                    end = randint(begin + min_seq_length * skip, dp['end'])
                    labels = build_batch(data_store, begin, end + 1, max_seq_length, True, increment=skip,
                                         truncate=truncate)
                    if iteration_count < 4 and min_standard_deviation is not None and np.std(labels,
                                                                                             axis=0).max() < min_standard_deviation:
                        continue
                    pick = randint(1, 100)
                    is_auto_encoder = False
                    if pick < 5:
                        funct = aug_auto_encoder
                        is_auto_encoder = True
                    elif pick < 10:
                        funct = aug_alternate_missing
                    elif pick < 25:
                        funct = aug_sentinals_missing
                    elif pick < 65:
                        funct = aug_kp_cluster_missing
                    elif pick < 75:
                        funct = aug_kp_missing
                    elif pick <= 100:
                        funct = aug_clusters_missing
                    inputs, mask = funct(deepcopy(labels), BLANK)
                    mask = np.array(mask) / len(inputs) * 100
                    r = sample(list(range(0, 360, randint(3, 10))), 1)[0]
                    inputs, labels = random_rigid_transformation(inputs, deepcopy(labels), rotation=r, min_x=min_x,
                                                                 min_y=min_y, max_x=max_x, max_y=max_y,
                                                                 static_scale=scale)
                    if randint(1, 5) == 1 or is_auto_encoder:
                        inputs = add_random_noise(inputs, random_noise, is_auto_encoder)
                    avg_length_list = avg_length_list + (len(inputs) - avg_length_list) / (count + 1)

                    while len(inputs) != max_seq_length:
                        inputs.append(PAD)
                        labels.append(PAD)
                    write = [inputs, labels]
                    batch_rows.append(write)
                    if len(batch_rows) > 1000:
                        writer.writerows(batch_rows)
                        batch_rows.clear()
                    avg_mask = (avg_mask * count + mask) / (count + 1)
                    count += 1
                    total_count += 1
            iteration_count += 1
    if len(batch_rows) > 0:
        writer.writerows(batch_rows)
    print('\n', avg_mask)
    print('average length', avg_length_list)


def plot_pose(pose, vectors, axes=None, alpha=1.0, original=False, limits=[[0, 1000]] * 3):
    '''
    Plots single pose in 3D. Used for visualizing augmented poses.
    Args:
        pose: Pose to be plotted. A list of 3D keypoints
        vectors: A list of lists containing two indices for drawing skeleton lines.
        axes: Matplotlib 3D axes. Can be none if plotting first pose.
        alpha: Alpha value set to the drawn pose
        original: Whether it is the  original augmented pose. It will be drawn in red.
        limits: X,Y, and Z limits for plots

    Returns: Axes to chain more plotting operations

    '''
    if axes is None:
        axes = plt.axes(projection='3d')
        axes.set_xlim(*limits[0])
        axes.set_ylim(*limits[1])
        axes.set_zlim(*limits[2])

    lines = []
    for v in vectors:
        px = np.array([pose[v[0]][0], pose[v[1]][0]])
        py = np.array([pose[v[0]][1], pose[v[1]][1]])
        pz = np.array([pose[v[0]][2], pose[v[1]][2]])
        lines.append([px, py, pz])
    for line in lines:
        if original:
            axes.plot(line[0], line[1], line[2], color='r', linewidth=1.5, alpha=alpha)
        else:
            axes.plot(line[0], line[1], line[2], color='b', linewidth=1.5, alpha=alpha)
    return axes
