#!/usr/bin/env python
import os, sys, glob
import os.path
import random
import logging
import open3d.ml.tf as ml3d
#import open3d.ml.torch as ml3d
import numpy as np
from sklearn.model_selection import train_test_split

import dotenv
env_file = dotenv.find_dotenv()
dotenv.load_dotenv(env_file)
parsed_dotenv = dotenv.dotenv_values()
formatted_data_dir = parsed_dotenv["formatted_data_dir"]

def get_data(pc_names, data_path, num_points=9000):

    pc_data = []
    for i, name in enumerate(pc_names):

        pc_path = os.path.join(data_path, name + '.bin')
        label_path = os.path.join(data_path, name + '.label')

        raw_data = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        if num_points > raw_data.shape[0]:
            logging.warning(f"Pointcloud has only {raw_data.shape[0]} points, which is fewer than num_points ({num_points})")
        sampled_data = np.array(random.sample(list(raw_data), num_points))

        points = sampled_data[:, 0:3]
        labels = sampled_data[:, -1].astype(np.int32)


        data = {
            'point': points,
            'feat': None,
            'label': labels,
        }
        pc_data.append(data)

    return pc_data


def get_kitti_data(pc_names, path):

    pc_data = []
    for i, name in enumerate(pc_names):
        pc_path = os.path.join(path, 'points', name + '.npy')
        label_path = os.path.join(path, 'labels', name + '.npy')
        point = np.load(pc_path)[:, 0:3]
        label = np.squeeze(np.load(label_path))

        data = {
            'point': point,
            'feat': None,
            'label': label,
        }
        pc_data.append(data)

    return pc_data




def pred_custom_data(pc_names, pcs, pipeline_r, pipeline_k):
    vis_points = []
    for i, data in enumerate(pcs):
        name = pc_names[i]

        results_r = pipeline_r.run_inference(data)
        pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
        # WARNING, THIS IS A HACK
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label_r[0] = 0

        results_k = pipeline_k.run_inference(data)
        pred_label_k = (results_k['predict_labels'] + 1).astype(np.int32)
        # WARNING, THIS IS A HACK
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label_k[0] = 0

        label = data['label']
        pts = data['point']

        vis_d = {
            "name": name,
            "points": pts,
            "labels": label,
            "pred": pred_label_k,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_randlanet",
            "points": pts,
            "labels": pred_label_r,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_kpconv",
            "points": pts,
            "labels": pred_label_k,
        }
        vis_points.append(vis_d)

    return vis_points





# ------------------------------


def main():
    season10_labels = {
        0: 'unlabeled',
        1: 'plant',
        2: 'soil'
    }
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(season10_labels.keys()):
        lut.add_label(season10_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    model = ml3d.models.RandLANet()
    pipeline_r = ml3d.pipelines.SemanticSegmentation(model)
    pipeline_r.load_ckpt(model.cfg.ckpt_path)

    model = ml3d.models.KPFCNN(in_radius=10)
    pipeline_k = ml3d.pipelines.SemanticSegmentation(model)
    pipeline_k.load_ckpt(model.cfg.ckpt_path)

    data_path = os.path.join(formatted_data_dir, 'season10_3D_labeled')
    point_cloud_ids = [os.path.basename(x).split('.')[0] for x in glob.glob(os.path.join(data_path, "*.label"))]

    training_ids, testing_ids = train_test_split(point_cloud_ids, train_size=0.75)

    if False:
        # Smaller for testing.
        training_ids = training_ids[0:2]

    pc_names = training_ids
    pcs = lpcs = get_data(pc_names, data_path)
    #pcs_with_pred = pred_custom_data(pc_names, pcs, pipeline_r, pipeline_k)

    if False:
        # Use Kitti data
        data_path = '/home/equant/work/repos/3d_semantic_segmentation/Open3D-ML/examples/demo_data'
        pc_names = ["000700", "000750"]
        pcs = kpcs = get_kitti_data(pc_names, data_path)

    pcs_with_pred = pred_custom_data(pc_names, pcs, pipeline_r, pipeline_k)

    v.visualize(pcs_with_pred)


if __name__ == "__main__":
    main()


"""
def do_KPFCNN(pc_names, pcs, pipeline_r, pipeline_k)
    model = ml3d.models.KPFCNN(in_radius=10)
    pipeline_k = ml3d.pipelines.SemanticSegmentation(model)
    pipeline_k.load_ckpt(model.cfg.ckpt_path)

def do_inference(pc_names, pcs, pipeline_r, pipeline_k)

    vis_points = []

    models = [
            'KPFCNN' : ml3d.models.KPFCNN(in_radius=10),
            'RandLANet' : ml3d.models.RandLANet(),
    ]

    for i, data in enumerate(pcs):
        name = pc_names[i]
        label = data['label']
        pts = data['point']

        for model in models:
            model = ml3d.models.RandLANet()
            pipeline = ml3d.pipelines.SemanticSegmentation(model)
            #pipeline.load_ckpt(model.cfg.ckpt_path)
            results = pipeline.run_inference(data)
            pred_label = (results['predict_labels'] + 1).astype(np.int32)
            # WARNING, THIS IS A HACK
            # Fill "unlabeled" value because predictions have no 0 values.
            pred_label[0] = 0
"""

