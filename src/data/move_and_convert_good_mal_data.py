import argparse, sys
import glob
import os.path
from pathlib import Path
import pdb
import random

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)

import json
import numpy as np
import pandas as pd
from pathlib import Path
import open3d as o3d

key_id_map_file = 'key_id_map.json'
meta_file       = 'meta.json'

def get_args():

    """
    If you are in jupyter/ipython you can do something like this...
In [8]: sys.argv = ['move_and_convert_good_mal_data.py', '--input', '/media/equant/7fe7f0a0-e17f-46d2-82
...: d3-e7a8c25200bb/work/raw_data/season_10_lettuce_yr_2020/level_3/scanner3DTop/MAL-3000_examples/a
...: uto_project', '-o', '/media/equant/7fe7f0a0-e17f-46d2-82d3-e7a8c25200bb/work/raw_data/season_10_
...: lettuce_yr_2020/level_3/scanner3DTop/MAL-3000_examples/auto_project-generic']
    """
    
    parser = argparse.ArgumentParser(
        description='Convert supervisely datasets to a generic format (for semantic segmentation).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i',
                        '--input',
                        help='The path to the directory that contains the Supervisely dataset',
                        metavar='str',
                        type=str,
                        required=True)

    parser.add_argument('-o',
                        '--output',
                        help='The output directory to which the resulting generic dataset will be saved. ',
                        metavar='str',
                        type=str,
                        required=True)

    return parser.parse_args()

def good_pcd(filepath, qdf):
    """
    qdf is the quality dataframe.  Says if file is good or bad
    filepath is the complete path to the json file within the supervisely dir-structure.
    """
    filename = os.path.basename(filepath)
    pcd_filename = filename[:-5]
    if pcd_filename not in qdf.index:
        log.error(f"Something is wrong.  Can't find {pcd_filename} in our quality dataframe")
        sys.exit(0)
    if qdf.loc[pcd_filename, 'MAL Quality'] == 'Good':
        return True
    else:
        return False


def main():

    args = get_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    if not os.path.isdir(args.input):
        log.error(f"Input directory doesn't exist: {args.input}")

    supervisely_data_path = args.input
    output_path   = args.output

    mal_quality_csv_path = os.path.join(supervisely_data_path, 'mal_quality.csv')
    if not os.path.isfile(mal_quality_csv_path):
        log.error(f"Can't find a mal_quality.csv file.  Did you run the GUI to rank the MAL data?")
        sys.exit(0)
    qdf = pd.read_csv(mal_quality_csv_path, index_col='pcd');

    with open(os.path.join(supervisely_data_path, key_id_map_file), "r") as read_file:
        key_id_map_dict = json.load(read_file)

    with open(os.path.join(supervisely_data_path, meta_file), "r") as read_file:
        meta_dict = json.load(read_file)

    label_names =  ['unlabeled']
    label_names += [x['title'] for x in meta_dict['classes']]

    Path(os.path.join(output_path)).mkdir(parents=True, exist_ok=True)

    data_dict = dict()


    # We will now loop through every file in the `ann` directory.  Within each
    # file, we must loop through the json data.  First, we extract the label info
    # from the `objects` data and then we loop through each partial pointcloud
    # within the `figures` data.

    count = 0
    all_ann_files = glob.glob(os.path.join(supervisely_data_path,'ds0','ann','*.pcd.json'))

    for f in all_ann_files:
        count=count+1
        print(f"Progress: {count}")
        if not good_pcd(f, qdf):
            continue
        with open(f, "r") as read_file:
            ann_dict = json.load(read_file)
        label_lookup = dict()
        filename = os.path.split(f)[-1][:-5]    # remove trailing .json
        training_id = filename[:-4]             # remove trailing .pcd
        pc_file = os.path.join(supervisely_data_path, 'ds0', 'pointcloud', filename)
        pc_data = np.asarray(o3d.io.read_point_cloud(pc_file).points).astype(np.float32)
        # Add a column to data so that we can insert our label
        labeled_data = np.zeros((len(pc_data),4))
        labeled_data[:,:-1] = pc_data         # Unlabeled data will be zero
        # Loop through all of the objects to get label names.
        for obj in ann_dict['objects']:
            if obj['key'] not in label_lookup.keys():
                label_lookup[obj['key']] = obj['classTitle']
            else:
                log.error(f"Found {obj['key']} in label_lookup.  That was unexpected")
        # Loop through all of the figures to get the indexes/labels
        for fig in ann_dict['figures']:
            if fig['objectKey'] in label_lookup:
                label = label_lookup[fig['objectKey']]
            else:
                log.error("Didn't find {fig['objectKey']} in label_lookup.  That shouldn't happen.")
            labeled_data[:,-1][fig['geometry']['indices']] = label_names.index(label)
        number_of_labels_found = len(np.unique(labeled_data[:,-1]))
        if number_of_labels_found > 1:
            # Save labeled data.
            data_dict[training_id] = labeled_data
        else:
            # Skip, because it's not properly labeled.
            log.warning(f"{training_id} has only {number_of_labels_found} labels")


    for training_id in data_dict.keys():
        print(f"Saving {training_id}")
        point_cloud_save_file = os.path.join(output_path, training_id+".bin")
        label_save_file = os.path.join(output_path, training_id+".label")
        #print(point_cloud_save_file)
        #print(label_save_file)
        data_dict[training_id].astype(np.float32).tofile(point_cloud_save_file)
        data_dict[training_id].astype(np.uint32).T[-1].tofile(label_save_file)


if __name__ == "__main__":
    main()
