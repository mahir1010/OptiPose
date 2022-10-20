import csv
import os
import random

import numpy as np
from scipy.spatial.transform import Rotation

from OptiPose import Skeleton, Part, DLTrecon, rotate, OptiPoseConfig
from OptiPose.data_store_interface import DataStoreInterface, initialize_datastore_reader
from OptiPose.video_reader_interface import BaseVideoReaderInterface, initialize_video_reader


def reconstruct_3D(config, views, data_readers: DataStoreInterface, name='recon', threshold=0.95):
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
    scale = 1000.0
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
            recon_data[name] = rotate(DLTrecon(3, len(subset), dlt_subset, subset), rotation_matrix, scale) + trans_mat
            prob_data[name] = min(subset, key=lambda x: x.likelihood).likelihood
        skeleton_3D = Skeleton(config['body_parts'], recon_data, prob_data)
        csv_writer.writerow(
            [skeleton_3D[part].tolist() if skeleton_3D[part] > threshold else None for part in
             config['body_parts']])
    print("\rReconstructing 3D Scene 100% complete")


def generate_EasyWand_data(config, csv_maps, static_points):
    print(csv_maps.keys())
    common_indices = [set(csv_maps[cam].data.index) for cam in config['views']]
    common_indices = list(common_indices[0].intersection(*common_indices[1:]))
    # remove = list(range(len(common_indices)-1,-1,-2))
    # for r in remove:
    #     common_indices.pop(r)
    writer = csv.writer(open(os.path.join(config['output_folder'], 'wandPoints.csv'), 'w'))
    parts = set(*config['skeleton'])
    parts = parts.difference(static_points)
    for idx in common_indices:
        if all(csv_df.get_marker(idx, name) > 0 for csv_df in csv_maps.values() for name in parts):
            builder = []
            for part in parts:
                for csv_df in csv_maps.values():
                    builder.extend(np.round(csv_df.get_marker(idx, part)[:2]).astype(np.int32))
            writer.writerow(builder)
    valid_static = set()
    for static_point in static_points:
        insert = True
        for csv_df in csv_maps.values():
            pt = csv_df.get_valid_marker(static_point)
            if pt is None:
                insert = False
        if insert:
            valid_static.add(static_point)
    writer1 = csv.writer(open(os.path.join(config['output_folder'], 'background.csv'), 'w'))
    print(valid_static)
    for valid in valid_static:
        builder = []
        for csv_df in csv_maps.values():
            builder.extend(np.round(csv_df.get_valid_marker(valid)[:2]).astype(np.int32))
        writer1.writerow(builder)


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
    for i in range(len(datastore)):
        if i % 100 == 0:
            print(f'\r{i}/{len(datastore)}', end='')
        for body_part in datastore.body_parts:
            if max(np.abs(velocity_datastore.get_marker(i, body_part)).tolist()) > threshold:
                datastore.delete_marker(i, body_part, True)
    datastore.save_file(f'{datastore.base_file_path}_velocity_filtered.csv')


if __name__ == '__main__':
    import yaml as yml

    config = yml.safe_load(open('/home/mahirp/Projects/Pose Annotator/Resources/220510_1.yaml', 'r'))
    view = config['views']['Blue']
    compute_invalid_frame_indices(initialize_video_reader(view['video_file'], 60, 'opencv'),
                                  initialize_datastore_reader(config['body_parts'], view['annotation_file'],
                                                              'deeplabcut'))
    view = config['views']['Pink']
    compute_invalid_frame_indices(initialize_video_reader(view['video_file'], 60, 'opencv'),
                                  initialize_datastore_reader(config['body_parts'], view['annotation_file'],
                                                              'deeplabcut'))
    view = config['views']['Yellow']
    compute_invalid_frame_indices(initialize_video_reader(view['video_file'], 60, 'opencv'),
                                  initialize_datastore_reader(config['body_parts'], view['annotation_file'],
                                                              'deeplabcut'))
    # csv_maps={}
    # for view in config['views']:
    #     csv_maps[view]=DeeplabcutDataStore(config['body_parts'],config['views'][view]['annotation_file'])
    # static_points=['top_left','top_right','bottom_left','bottom_right']
    # generate_EasyWand_data(config, csv_maps, static_points)
