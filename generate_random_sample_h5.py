#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2021-02-24
Purpose: Generate randomly-sampled H5 files from PLY point cloud data.
"""

import argparse
import os
import sys
import random
import glob
import h5py
import numpy as np
import random
from plyfile import PlyData, PlyElement


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='PLY to H5 with random sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dir',
                        metavar='dir',
                        help='Directory containing PLY files')

    parser.add_argument('-n',
                        '--num_points',
                        help='Number of points to randomly sample from each point cloud',
                        metavar='num_points',
                        type=int,
                        default=2048)

    parser.add_argument('-o',
                        '--outdir',
                        help='Output directory',
                        metavar='outdir',
                        type=str,
                        default='ply2h5_out')

    parser.add_argument('-tp',
                        '--train_percentage',
                        help='Percentage of data to use for training (decimal)',
                        metavar='train_percentage',
                        type=float,
                        default=0.8)

    # parser.add_argument('-o',
    #                     '--on',
    #                     help='A boolean flag',
    #                     action='store_true')

    return parser.parse_args()


# --------------------------------------------------
def get_paths(directory):
    ortho_list = []

    for root, dirs, files in os.walk(directory):
        for name in files:
            if '.npy' in name:
                ortho_list.append(os.path.join(root, name))

    if not ortho_list:

        raise Exception(f'ERROR: No compatible images found in {directory}.')


    print(f'Pointclouds to process: {len(ortho_list)}')

    return ortho_list


# --------------------------------------------------
def sub_sample(filename, num_points):
    full_data = np.load(filename).tolist()
    sampled_data = np.array(random.sample(full_data, num_points))
    sampled_data.shape
    data = sampled_data[:, :3]
    label = np.array(sampled_data[:, 3:4].ravel(), dtype='uint8')

    return data, label


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()
    random.seed(10)
    cnt = 0

    file_list = get_paths(args.dir)
    random.shuffle(file_list)
    train, test = np.split(file_list, [int(len(file_list)*args.train_percentage)])

    for ml_set in [train, test]:
        cnt += 1
        set_name = 'train' if cnt==1 else 'test'

        if not os.path.isdir(os.path.join(args.outdir, set_name)):
            os.makedirs(os.path.join(args.outdir, set_name))

        agg_data = np.zeros((len(ml_set), args.num_points, 3))
        agg_label = np.zeros((len(ml_set), args.num_points), dtype = np.uint8)

        for i in range(0, len(ml_set)):

            basename = os.path.splitext(os.path.basename(ml_set[i]))[0]
            data, label = sub_sample(ml_set[i], args.num_points)
            if len(np.unique(label)) == 3:
                agg_data[i] = data
                agg_label[i] = label

        out_path = os.path.join(args.outdir, set_name, f'{set_name}_with_labels.h5')
        f = h5py.File(out_path, 'w')
        data = f.create_dataset("data", data = agg_data)
        pid = f.create_dataset("label", data = agg_label)


# --------------------------------------------------
if __name__ == '__main__':
    main()
