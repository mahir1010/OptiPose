import csv
import os
from random import random

import numpy as np
from scipy.spatial.transform import Rotation

from OptiPose import Skeleton, Part, DLTrecon, rotate, OptiPoseConfig
from OptiPose.data_store_interface import DataStoreInterface, DeeplabcutDataStore
from OptiPose.post_processor_interface import ClusterAnalysisProcess
from OptiPose.utils import compute_distance_matrix, magnitude
from OptiPose.video_reader_interface import BaseVideoReaderInterface


def reconstruct_3D(config: OptiPoseConfig, views, data_readers: DataStoreInterface, name='recon', threshold=0.95):
    dlt_coefficients = np.array([camera['dlt_coefficients'] for camera in views.values()])
    reconstruction_algorithm = config['OptiPose']['reconstruction_algorithm']
    origin_2D = [camera['axes']['origin'] for camera in views.values()]
    x_max_2D = [camera['axes']['x_max'] for camera in views.values()]
    y_max_2D = [camera['axes']['y_max'] for camera in views.values()]
    origin = Part((DLTrecon(3, len(origin_2D), dlt_coefficients, origin_2D)), "origin", 1)
    x_max = Part((DLTrecon(3, len(x_max_2D), dlt_coefficients, x_max_2D)), "x_max", 1)
    y_max = Part((DLTrecon(3, len(y_max_2D), dlt_coefficients, y_max_2D)), "y_max", 1)
    rotation_matrix = Rotation.align_vectors([x_max - origin, y_max - origin], [[1, 0, 0], [0, 1, 0]])[0].as_matrix()
    origin = Part(rotate(DLTrecon(3, len(origin_2D), dlt_coefficients, origin_2D), rotation_matrix), "origin", 1)
    scale = config.scale
    trans_mat = scale * -origin
    file = open(os.path.join(config['output_folder'],
                             f"{name}_{threshold}_{'_'.join([camera for camera in views]) if reconstruction_algorithm != 'auto_subset' else 'auto_subset'}.csv"),
                "w")
    csv_writer = csv.writer(file, delimiter=';')
    csv_writer.writerow(config['body_parts'])
    length = len(min(data_readers, key=lambda x: len(x)))
    for iterator in range(length):
        print("\rReconstructing 3D Scene ", round(iterator / length * 100), "% complete ", end='')
        skeleton_2D = [reader.get_skeleton(iterator) for reader in data_readers]
        recon_data = {}
        prob_data = {}
        for name in config['body_parts']:
            subset = [sk[name] for sk in skeleton_2D]
            dlt_subset = dlt_coefficients
            if reconstruction_algorithm == "auto_subset":
                indices = [subset[i].likelihood >= threshold for i in range(len(subset))]
                if sum(indices) >= 2:
                    dlt_subset = dlt_subset[indices, :]
                    subset = [element for i, element in enumerate(subset) if indices[i]]
                    recon_data[name] = rotate(DLTrecon(3, len(subset), dlt_subset, subset), rotation_matrix,
                                              scale) + trans_mat
                    prob_data[name] = min(subset, key=lambda x: x.likelihood).likelihood
        skeleton_3D = Skeleton(config['body_parts'], recon_data, prob_data)
        csv_writer.writerow(
            [skeleton_3D[part].tolist() if skeleton_3D[part] > threshold else None for part in
             config['body_parts']])
    print("\rReconstructing 3D Scene 100% complete")


def generate_EasyWand_data(config: OptiPoseConfig, csv_maps, common_indices, static_points_map):
    file = open(os.path.generate_distance_matricesjoin(config.output_folder,
                                                       f'{config.project_name}_calibration_camera_order.txt'), 'w')
    camera_profile = open(os.path.join(config.output_folder, f'{config.project_name}_calibration_camera_profiles.txt'),
                          'w')
    for i, camera in enumerate(config.views):
        file.write(f'{camera} ')
        camera_config = config.views[camera]
        profile = f'{i + 1} {camera_config.f_px} {camera_config.resolution[0]} {camera_config.resolution[1]} {camera_config.principal_point[0]} {camera_config.principal_point[1]} 1 0 0 0 0 0\n'
        camera_profile.write(profile)
    camera_profile.close()
    file.close()
    static_writer = csv.writer(open(os.path.join(config.output_folder, f'{config.project_name}_background.csv'), 'w'))
    writer = csv.writer(open(os.path.join(config.output_folder, f'{config.project_name}_WandPoints.csv'), 'w'))
    parts = config.body_parts
    for idx in common_indices:
        builder = []
        for part in parts:
            for csv_df in csv_maps.values():
                builder.extend(np.round(csv_df.get_part(idx, part)[:2]).astype(np.int32))
        writer.writerow(builder)

    static_rows = []
    for camera in static_points_map:
        for i, point in enumerate(static_points_map[camera]):
            if len(static_rows) == i:
                static_rows.append([])
            static_rows[i].extend(point)
    static_writer.writerows(static_rows)


def compute_invalid_frame_indices(video_reader: BaseVideoReaderInterface, data_store: DataStoreInterface,
                                  sample_size=8):
    writer = csv.writer(open(f'{video_reader.base_file_path}_frame_status.csv', 'w'))
    writer.writerow(['isValid'])
    writer.writerow([1])
    previous_frame = video_reader.next_frame()
    current_frame = video_reader.next_frame()
    while current_frame is not None:
        print(f'\r{video_reader.get_current_index()}/{video_reader.get_number_of_frames()}', end='')
        index = video_reader.get_current_index()
        x = random.sample(range(current_frame.shape[0]), sample_size)
        y = random.sample(range(current_frame.shape[1]), sample_size)
        if not np.all(previous_frame[x, y] == current_frame[x, y]):
            writer.writerow([1])
        else:
            writer.writerow([0])
            data_store.delete_skeleton(index)
        previous_frame = current_frame
        current_frame = video_reader.next_frame()
    data_store.save_file(f'{data_store.base_file_path}_valid_frames.csv')


def find_equal_frame(video_reader, reference_frame, sample_size=10):
    x = random.sample(range(reference_frame.shape[0]), sample_size)
    y = random.sample(range(reference_frame.shape[1]), sample_size)
    reference_pixels = reference_frame[x, y]
    current_frame = video_reader.next_frame()
    while current_frame is not None:
        print(f'\r{video_reader.get_current_index()}', end='')
        if np.all(current_frame[x, y] == reference_pixels):
            print(video_reader.get_current_index())
            return video_reader.get_current_index()
        current_frame = video_reader.next_frame()
    return -1


def filter_invalid_frames(data_reader: DataStoreInterface, invalid_indices):
    for index, skeleton in data_reader.row_iterator():
        print(f'\r{index}/{len(data_reader)}', end='')
        if not invalid_indices.loc[index, 'isValid']:
            for part in data_reader.body_parts:
                skeleton[part].likelihood = 0.0
            data_reader.set_skeleton(index, skeleton, True)
    data_reader.save_file('test.csv')


def update_alignment_matrices(config: OptiPoseConfig, source_views: list):
    if len(source_views) < 2:
        return False
    try:
        dlt_coefficients = np.array([config.views[view].dlt_coefficients for view in source_views])
        origin_2D = [config.views[view].axes['origin'] for view in source_views]
        x_max_2D = [config.views[view].axes['x_max'] for view in source_views]
        y_max_2D = [config.views[view].axes['y_max'] for view in source_views]
        origin = Part((DLTrecon(3, len(origin_2D), dlt_coefficients, origin_2D)), "origin", 1)
        x_max = Part((DLTrecon(3, len(x_max_2D), dlt_coefficients, x_max_2D)), "x_max", 1)
        y_max = Part((DLTrecon(3, len(y_max_2D), dlt_coefficients, y_max_2D)), "y_max", 1)
        rotation_matrix = Rotation.align_vectors([x_max - origin, y_max - origin], [[1, 0, 0], [0, 1, 0]])[
            0].as_matrix()
        origin = Part(rotate(DLTrecon(3, len(origin_2D), dlt_coefficients, origin_2D), rotation_matrix), "origin", 1)
        x_max = Part(rotate(DLTrecon(3, len(x_max_2D), dlt_coefficients, x_max_2D), rotation_matrix), "x_max", 1)
        y_max = Part(rotate(DLTrecon(3, len(y_max_2D), dlt_coefficients, y_max_2D), rotation_matrix), "y_max", 1)
        config.computed_scale = (config.scale / magnitude(x_max - origin) + config.scale / magnitude(
            y_max - origin)) / 2
        trans_mat = -origin
        config.rotation_matrix = rotation_matrix
        config.translation_matrix = trans_mat
    except:
        return False
    return True


def convert_data_flavor(source: DataStoreInterface, target: DataStoreInterface):
    assert not os.path.exists(target.path)
    writer = csv.writer(open(target.path, 'w'), delimiter=target.SEP)
    writer.writerows(target.get_header_rows())
    for index, skeleton in source.row_iterator():
        if index % 200 == 0:
            print(f'\r{index}/{len(source)}', end='')
        writer.writerow(target.convert_to_list(index, skeleton))


def translate_data_store(source: DataStoreInterface, translation_vector: np.array, threshold=0.7):
    writer = csv.writer(open(f'{source.base_file_path}_translated.csv', 'w'), delimiter=source.SEP)
    writer.writerows(source.get_header_rows())
    for index, sk in source.row_iterator():
        if index % 500 == 0:
            print(f'\r{index}/{len(source)}', end='')
        for part in sk:
            if part > threshold:
                sk[part.name] = part + translation_vector
        sk = source.convert_to_list(index, sk)
        writer.writerow(sk)


def remove_duplicate_rows(source: DataStoreInterface):
    previous_row = None
    removed = []
    for index, row in source.row_iterator():
        if index % 200 == 0:
            print(f'\r{index}/{len(source)}', end='')
        if previous_row is not None and row == previous_row:
            removed.append(index)
            source.delete_skeleton(index)
        else:
            previous_row = row
    source.save_file(f'{source.base_file_path}_de_duplicated.csv')
    return removed


def filter_from_velocity(datastore: DataStoreInterface, velocity_datastore: DataStoreInterface, threshold):
    counts = 0
    for i in range(len(datastore)):
        if i % 100 == 0:
            print(f'\r{i}/{len(datastore)}', end='')
        for body_part in datastore.body_parts:
            if magnitude(velocity_datastore.get_part(i, body_part)) > threshold:
                datastore.delete_part(i, body_part, True)
                counts += 1
    print(f"\n{counts} data points filtered")
    datastore.save_file(f'{datastore.base_file_path}_velocity_filtered.csv')


def generate_distance_matrices(config: OptiPoseConfig, data_points: list):
    output_distance_matrices = np.zeros((2, config.num_parts, config.num_parts))
    distance_matrices = []
    for point in data_points:
        distance_matrices.append(compute_distance_matrix(point))
    output_distance_matrices[0] = np.mean(distance_matrices, axis=0)
    output_distance_matrices[1] = np.std(distance_matrices, axis=0)
    return output_distance_matrices


def pick_calibration_candidates(config: OptiPoseConfig, data_stores: list, resolution, bin_size):
    assert len(data_stores) > 1
    cluster_analysis = ClusterAnalysisProcess(config.threshold)
    cluster_analysis.PRINT = True
    for i in range(len(data_stores)):
        if not data_stores[i].verify_stats():
            cluster_analysis.process(data_stores[i])
            data_stores[i] = cluster_analysis.get_output()
    accurate_data_points = data_stores[0].stats.accurate_data_points
    for data_store in data_stores[1:]:
        accurate_data_points = data_store.stats.intersect_accurate_data_points(accurate_data_points)
    num_bins = (resolution[0] // bin_size, resolution[1] // bin_size)
    spatial_bins = np.zeros(num_bins)
    frame_number = np.zeros(num_bins)
    part = data_stores[0].body_parts[0]
    candidates = []
    for accurate_data in accurate_data_points:
        for index in range(accurate_data['begin'], accurate_data['end']):
            position = data_stores[0].get_part(index, part)
            x_bin, y_bin = int(position[0] / bin_size), int(position[1] / bin_size)
            if 0 <= x_bin < num_bins[0] and 0 <= y_bin < num_bins[1] and (
                    spatial_bins[x_bin][y_bin] < 3 and index - frame_number[x_bin][y_bin] > 10):
                spatial_bins[x_bin][y_bin] += 1
                frame_number[x_bin][y_bin] = index
                candidates.append(index)
    return candidates


def refactor_likelihood_for_uncertain_region(data_store: DeeplabcutDataStore, uncertainty_regions: list):
    for index, skeleton in data_store.row_iterator():
        if index % 100 == 0:
            print(f'\r{index}/{len(data_store)}', end='')
        for part in data_store.body_parts:
            for uncertainty_region in uncertainty_regions:
                if uncertainty_region[0][0] < skeleton[part][0] < uncertainty_region[0][1] and uncertainty_region[1][
                    0] < skeleton[part][1] < uncertainty_region[1][1]:
                    data_store.delete_part(index, part, force_remove=True)
                    break
    data_store.save_file(data_store.path.replace('.csv', '_region_filtered.csv'))
