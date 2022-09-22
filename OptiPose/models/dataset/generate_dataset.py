import csv
import os

from OptiPose.models.dataset.augmentation import *
from OptiPose.post_processor_interface import ClusterAnalysisProcess

TEST_FILE_SIZE = 6500


def generate_dataset(root_path, dataset_name, is_test, data_stores: list, total_samples, max_seq_length=60,
                     min_seq_length=2, prefix="", suffix="", min_x=-100, max_x=1100, min_y=-100, max_y=1100, scale=1.0):
    assert 2 < min_seq_length < max_seq_length
    file_name = f"{prefix}_{dataset_name}_{suffix}_{max_seq_length}_{total_samples}{'_test' if is_test else ''}.csv"
    samples_per_file = total_samples // len(data_stores) if not is_test else (TEST_FILE_SIZE // len(data_stores))
    writer = csv.writer(open(os.path.join(root_path, file_name), 'w'), delimiter='|')
    writer.writerow(['input', 'label'])
    BLANK = [MAGIC_NUMBER] * 3
    PAD_VAL = [0, 0, 0]
    PAD = [PAD_VAL.copy() for p in data_stores[0].body_parts]

    body_parts = data_stores[0].body_parts.copy()

    avg_length_list = [min_seq_length]
    analysis_processor = ClusterAnalysisProcess()
    analysis_processor.PRINT = True
    batch_rows = []
    avg_mask = np.array([0] * len(body_parts))
    for data_store in data_stores:
        print(data_store.path)
        if not data_store.verify_stats():
            analysis_processor.process(data_store)
            data_store = analysis_processor.get_output()
        count = 0
        flag = True
        while flag:
            for dp in data_store.stats.iter_accurate_clusters():
                if count >= samples_per_file:
                    flag = False
                    break
                if count % 100 == 0:
                    print(f'\r {count}/{len(data_stores) * samples_per_file}', end='')
                if dp['end'] - dp['begin'] > min_seq_length:
                    begin = randint(dp['begin'], dp['end'] - min_seq_length)
                    end = randint(begin + min_seq_length, dp['end'])
                    labels = []
                    for i in range(begin, end + 1):
                        if len(labels) >= max_seq_length:
                            break
                        labels.append(data_store.get_numpy(i))
                        assert not np.any(np.all(labels[-1] == BLANK, axis=1))
                        labels[-1] = labels[-1].tolist()
                    pick = randint(1, 100)
                    if pick < 10:
                        funct = aug_auto_encoder
                    if pick < 40:
                        funct = aug_clusters_missing
                    elif pick < 70:
                        funct = aug_kp_missing
                    elif pick <= 100:
                        funct = aug_kp_cluster_missing
                    inputs, mask = funct(labels.copy(), BLANK)
                    mask = np.array(mask) / len(inputs) * 100
                    r = sample(list(range(0, 360, randint(3, 10))), 1)[0]
                    inputs, labels = random_rigid_transformation(inputs, labels.copy(), rotation=r, min_x=min_x,
                                                                 min_y=min_y, max_x=max_x, max_y=max_y,
                                                                 static_scale=scale)
                    avg_length_list.append(len(inputs))
                    if len(avg_length_list) == 100:
                        avg_length_list.pop(0)

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
    if len(batch_rows) > 0:
        writer.writerows(batch_rows)
    print('\n', avg_mask)
