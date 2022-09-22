from OptiPose.video_reader_interface.VideoReaderInterface import BaseVideoReaderInterface

video_readers = {}
try:
    from OptiPose.video_reader_interface.CV2VideoInterface import CV2VideoReader

    video_readers[CV2VideoReader.FLAVOR] = CV2VideoReader
except:
    pass
try:
    from OptiPose.video_reader_interface.DeffcodeVideoInterface import DeffcodeVideoReader

    video_readers[DeffcodeVideoReader.FLAVOR] = DeffcodeVideoReader
except:
    pass


def initialize_video_reader(video_path, fps, reader_type):
    try:
        return video_readers[reader_type](video_path, fps)
    except KeyError:
        raise Exception(f"{reader_type} flavor is not installed.")
    except Exception as e:
        raise Exception(f"Error while initializing video reader ({reader_type})" + str(e))
    return None
