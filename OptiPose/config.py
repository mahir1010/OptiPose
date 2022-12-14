import os

import numpy as np
import yaml as yml

MAGIC_NUMBER = -4668
DEFAULT_THRESHOLD = 0.6


class CameraViews:
    def __init__(self, data_dictionary, framerate):
        self.axes = data_dictionary.get('axes', {})
        self.dlt_coefficients = np.array(data_dictionary.get('dlt_coefficients', []))
        assert self.dlt_coefficients.shape == (12,)
        self.framerate = framerate
        self.pos = np.array(data_dictionary.get('pos', []))
        self.resolution = np.array(data_dictionary.get('resolution', []))
        self.center = np.array(data_dictionary.get('center', []))
        self.f = data_dictionary.get('f', -1)

    def export_dict(self):
        return {
            'axes': self.axes,
            'dlt_coefficients': self.dlt_coefficients.tolist(),
            'pos': self.pos.tolist(),
            'resolution': self.resolution.tolist(),
            'f': self.f,
            'center': self.center.tolist()
        }


class OptiPoseConfig:
    ENABLE_FLOW_STYLE = ['name', 'output_folder', 'threshold', 'reprojection_toolbox', 'behaviours', 'body_parts',
                         'skeleton']

    def __init__(self, path):
        self.data_dictionary = yml.safe_load(open(path, 'r'))
        assert 0 <= DEFAULT_THRESHOLD < 1.0
        self.project_name = self.data_dictionary.get('name', 'unnamed')
        self.output_folder = self.data_dictionary['output_folder']
        assert os.path.exists(self.output_folder)
        self.threshold = float(self.data_dictionary['OptiPose'].get('threshold', DEFAULT_THRESHOLD))
        self.body_parts = self.data_dictionary['body_parts']
        self.num_parts = len(self.body_parts)
        self.skeleton = self.data_dictionary['skeleton']
        self.colors = list(self.data_dictionary.get('colors', []))
        self.framerate = self.data_dictionary['OptiPose']['framerate']
        self.views = {}
        if 'views' in self.data_dictionary:
            for view in self.data_dictionary['views']:
                self.views[view] = CameraViews(self.data_dictionary['views'][view], self.framerate)
        self.rotation_matrix = np.array(self.data_dictionary['OptiPose'].get('rotation_matrix', np.identity(3)),
                                        dtype=np.float)
        assert self.rotation_matrix.shape == (3, 3)
        self.scale = float(self.data_dictionary['OptiPose'].get('scale', 1.0))
        self.translation_matrix = np.array(self.data_dictionary['OptiPose'].get('translation_matrix', [0, 0, 0]),
                                           dtype=np.float) * self.scale
        self.reconstruction_algorithm = self.data_dictionary.get('reconstruction_algorithm', 'default')

    def export_dict(self):
        return {'name': self.project_name,
                'output_folder': self.output_folder,
                'body_parts': self.body_parts,
                'skeleton': self.skeleton,
                'colors': self.colors,
                'views': {view: self.views[view].export_dict() for view in self.views},
                'OptiPose': {
                    'threshold': self.threshold,
                    'scale': self.scale,
                    'framerate': self.framerate,
                    'rotation_matrix': self.rotation_matrix.tolist(),
                    'translation_matrix': self.translation_matrix.tolist(),
                    'reconstruction_algorithm': self.reconstruction_algorithm
                }
                }


def save_config(path, data_dict):
    out_file = ''
    for key in data_dict:
        if key in OptiPoseConfig.ENABLE_FLOW_STYLE:
            out_file += yml.dump({key: data_dict[key]}) + '\n'
        else:
            out_file += yml.dump({key: data_dict[key]}, default_flow_style=None) + '\n'
    save_file = open(path, 'w')
    save_file.write(out_file)
