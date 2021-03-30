from os.path import join, exists, dirname
import os, sys, glob
import pdb
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import random

from open3d.ml.torch.datasets import Custom3D
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
    #class_weights = [
                     #55437630, 320797, 541736, 2578735, 3274484, 552662, 184064,
                     #78858, 240942562, 17294618, 170599734, 6369672, 230413074,
                     #101130274, 476491114, 9833174, 129609852, 4506626, 1168181
	#]

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'unlabeled',
            1: 'plant',
            2: 'soil'
        }
        return label_to_names

    def get_split(self, split):
        return SuperviselySplit(self, split=split)

    def get_split_list(self, split):
        """Returns a dataset split.
        
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
			
		Raises:
			ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
    """
        cfg = self.cfg
        dataset_path = cfg.dataset_path
        file_list = []

        if split in ['train', 'training']:
            seq_list = cfg.training_split
        elif split in ['test', 'testing']:
            seq_list = cfg.test_split
        elif split in ['val', 'validation']:
            seq_list = cfg.validation_split
        elif split in ['all']:
            seq_list = cfg.all_split
        else:
            raise ValueError("Invalid split {}".format(split))

        for seq_id in seq_list:
            pc_path = join(dataset_path, f"{seq_id}.bin")
            file_list.append(pc_path)
            #file_list.append(
                #[join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        #pdb.set_trace()
        #file_list = np.concatenate(file_list, axis=0)

        return file_list



class SuperviselySplit():

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        #self.remap_lut_val = dataset.remap_lut_val


        if split == 'test':
            dataset.test_list = path_list

        log.info("FFFFound {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

        #pdb.set_trace()

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        #points = DataProcessing.load_pc_kitti(pc_path)

        raw_data = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)

        if 'resample_n' in self.cfg.keys():
            resample_n = self.cfg['resample_n']
            if resample_n < raw_data.shape[0]:
                sampled_data = np.array(random.sample(list(raw_data), resample_n))
            else:
                logging.warning(f"Pointcloud has only {raw_data.shape[0]} points, which is fewer than resample_n ({resample_n})")
                sampled_data = raw_data
        else:
            sampled_data = raw_data


        points = sampled_data[:, 0:3]
        labels = sampled_data[:, -1].astype(np.int32)

        #pdb.set_trace()

        data = {
            'point': points,
            'feat': None,
            'label': labels,
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.bin', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}

        return attr
