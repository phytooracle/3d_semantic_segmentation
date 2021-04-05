import glob
import os.path
from pathlib import Path
import pdb
import random
import logging
logging.basicConfig(
    level=logging.INFO,
    #format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)

import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path
#import random
#from plyfile import PlyData, PlyElement
import open3d as o3d

# Deal with local machine paths, etc.
import dotenv
env_file = dotenv.find_dotenv()
dotenv.load_dotenv(env_file)
parsed_dotenv = dotenv.dotenv_values()
project_dir = os.path.dirname(env_file)
raw_data_dir = parsed_dotenv["raw_data_dir"]
formatted_data_dir = parsed_dotenv["formatted_data_dir"]

data_path = os.path.join(raw_data_dir, "lettuce_semantic_segmentation")
key_id_map_file = 'key_id_map.json'
meta_file       = 'meta.json'
output_path     = os.path.join(formatted_data_dir, "converted_to_pointnet_dataset")

with open(os.path.join(data_path, key_id_map_file), "r") as read_file:
    key_id_map_dict = json.load(read_file)

with open(os.path.join(data_path, meta_file), "r") as read_file:
    meta_dict = json.load(read_file)

label_names =  ['unlabeled']
label_names += [x['title'] for x in meta_dict['classes']]

Path(os.path.join(output_path)).mkdir(parents=True, exist_ok=True)

data_dict    = dict()

# We will now loop through every file in the `ann` directory.  Within each
# file, we must loop through the json data.  First, we extract the label info
# from the `objects` data and then we loop through each partial pointcloud
# within the `figures` data.

all_ann_files = glob.glob(os.path.join(data_path,'ds0','ann','*.pcd.json'))

for f in all_ann_files:
#for f in ['/home/equant/work/sandbox/3D/open3dml/lettuce_semantic_segmentation/ds0/ann/train_10.pcd.json']:
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
        log.info(f"{training_id} has only {number_of_labels_found} labels")


# We now have a dictionary (data_dict) that contains each original pointcloud (used for labeling).  We want to loop
# through it and subset each pointcloud by labels.  In otherwords, we'll have a pointcloud for soil, and a pointcloud
# for plant, and a pointcloud for unlabeled.
#
# We will save data as npy files, and also as h5

from sklearn.model_selection import train_test_split
training_ids, testing_ids = train_test_split(list(data_dict.keys()), train_size=0.75)

num_points = 2048
training_agg_data  = np.zeros((len(training_ids), num_points, 3))
training_agg_label = np.zeros((len(training_ids)), dtype = np.uint8)
testing_agg_data   = np.zeros((len(testing_ids),  num_points, 3))
testing_agg_label  = np.zeros((len(testing_ids)), dtype = np.uint8)

for original_pc_name, original_pc_data in data_dict.items():
    print(original_pc_name)
    label_idx_found_in_original_pc = np.unique(original_pc_data[:,3]).astype(int)
    for label_idx in label_idx_found_in_original_pc:
        pc = original_pc_data[np.where(original_pc_data[:,3] == label_idx)]
        npy_save_file = os.path.join(output_path, f"{original_pc_name}_{label_names[label_idx]}")

        # npy
        np.save(npy_save_file, pc)
        if label_idx != 0:
            sampled_data = np.array(random.sample(pc[:,0:3].tolist(), num_points))
            if original_pc_name in training_ids:
                i = training_ids.index(original_pc_name)
                training_agg_data[i]  = sampled_data
                training_agg_label[i] = label_idx
            else:
                i = testing_ids.index(original_pc_name)
                testing_agg_data[i]  = sampled_data
                testing_agg_label[i] = label_idx

# H5

#f = h5py.File(out_path, 'w')
#data = f.create_dataset("data", data = pc[:,0:3])
#pid = f.create_dataset("label", data = pc[:,3])

Path(os.path.join(output_path,"train")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_path,"test")).mkdir(parents=True, exist_ok=True)
h5_training_file = os.path.join(output_path, "train","data_with_labels.h5")
h5_testing_file  = os.path.join(output_path, "test","data_with_labels.h5")
h5_training = h5py.File(h5_training_file, 'w')
h5_testing  = h5py.File(h5_testing_file, 'w')
 
data = h5_training.create_dataset("data",  data = training_agg_data)
pid  = h5_training.create_dataset("label", data = training_agg_label)
data = h5_testing.create_dataset("data",   data = testing_agg_data)
pid  = h5_testing.create_dataset("label",  data = testing_agg_label)

"""
useful debugging code...

print(len( data_dict[filename]['soil_pixel'] ))
print(len( data_dict[filename]['plant_pixel'] ))
print(len( data_dict[filename]['soil_pixel'] )+len( data_dict[filename]['plant_pixel'] ))
print(np.min(data_dict[filename]['plant_pixel']+ data_dict[filename]['soil_pixel']))
print(np.max(data_dict[filename]['plant_pixel']+ data_dict[filename]['soil_pixel']))
print(data.shape)
"""
