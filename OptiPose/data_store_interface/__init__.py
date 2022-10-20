from OptiPose.data_store_interface.DataStoreInterface import DataStoreInterface, DataStoreStats
from OptiPose.data_store_interface.DeeplabcutDataStore import DeeplabcutDataStore
from OptiPose.data_store_interface.FlattenedDataStore import FlattenedDataStore
from OptiPose.data_store_interface.OptiPoseDataStore3D import OptiPoseDataStore3D

datastore_readers = {OptiPoseDataStore3D.FLAVOR: OptiPoseDataStore3D, DeeplabcutDataStore.FLAVOR: DeeplabcutDataStore,
                     FlattenedDataStore.FLAVOR: FlattenedDataStore}


def initialize_datastore_reader(body_parts, path, reader_type):
    try:
        reader = datastore_readers[reader_type]
        return reader(body_parts, path)
    except Exception as e:
        raise Exception(f"Potentially incorrect reader type selected ({reader_type})" + str(e))
    return None
