from os.path import join, exists, dirname
import os, sys, glob
import pdb
import numpy as np
import pandas as pd
from pathlib import Path
import logging

#from open3d.ml.torch.datasets import Custom3D
from open3d.ml.tf.datasets import Custom3D
#from open3d.ml.utils import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class Supervisely(Custom3D):
    """
    https://app.supervise.ly/labeling/jobs/list
    """

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'unlabeled',
            1: 'PLANT!',
            2: 'ground',
        }
        return label_to_names

"""
class SuperviselySplit():

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        #self.remap_lut_val = dataset.remap_lut_val

        if split == 'test':
            dataset.test_list = path_list

        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        points = DataProcessing.load_pc_kitti(pc_path)

        dir, file = split(pc_path)
        label_path = join(dir, '../labels', file[:-4] + '.label')
        if not exists(label_path):
            labels = np.zeros(np.shape(points)[0], dtype=np.int32)
            if self.split not in ['test', 'all']:
                raise FileNotFoundError(f' Label file {label_path} not found')

        else:
            labels = DataProcessing.load_label_kitti(
                label_path, self.remap_lut_val).astype(np.int32)

        data = {
            'point': points[:, 0:3],
            'feat': None,
            'label': labels,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        dir, file = split(pc_path)
        _, seq = split(split(dir)[0])
        name = '{}_{}'.format(seq, file[:-4])

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr

"""
