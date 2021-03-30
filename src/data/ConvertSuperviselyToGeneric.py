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

import dotenv
env_file = dotenv.find_dotenv()
dotenv.load_dotenv(env_file)
parsed_dotenv = dotenv.dotenv_values()
raw_data_dir = parsed_dotenv["raw_data_dir"]
formatted_data_dir = parsed_dotenv["formatted_data_dir"]

raw_data_path       = os.path.join(raw_data_dir, 'lettuce_semantic_segmentation')
key_id_map_file = 'key_id_map.json'
meta_file       = 'meta.json'
output_path     = os.path.join(formatted_data_dir, "season10_3D_labeled")

with open(os.path.join(raw_data_path, key_id_map_file), "r") as read_file:
    key_id_map_dict = json.load(read_file)

with open(os.path.join(raw_data_path, meta_file), "r") as read_file:
    meta_dict = json.load(read_file)

label_names =  ['unlabeled']
label_names += [x['title'] for x in meta_dict['classes']]

Path(os.path.join(output_path)).mkdir(parents=True, exist_ok=True)

data_dict = dict()


# We will now loop through every file in the `ann` directory.  Within each
# file, we must loop through the json data.  First, we extract the label info
# from the `objects` data and then we loop through each partial pointcloud
# within the `figures` data.

all_ann_files = glob.glob(os.path.join(data_path,'ds0','ann','*.pcd.json'))

for f in all_ann_files:
    with open(f, "r") as read_file:
        ann_dict = json.load(read_file)
    label_lookup = dict()
    filename = os.path.split(f)[-1][:-5]    # remove trailing .json
    training_id = filename[:-4]             # remove trailing .pcd
    pc_file = os.path.join(data_path, 'ds0', 'pointcloud', filename)
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

