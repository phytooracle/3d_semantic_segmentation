#!/usr/bin/env python
import os, sys, glob
import os.path
import random
import logging
import pprint
import open3d.ml as _ml3d
import open3d.ml.tf as ml3d
#import open3d.ml.torch as ml3d
import numpy as np
from sklearn.model_selection import train_test_split

from supervisely_dataset import Supervisely

USE_KITTI = False #sometimes we want to test with a the KITTI dataset.
USE_SMALL_TESTING_SET = True

import dotenv
env_file = dotenv.find_dotenv()
dotenv.load_dotenv(env_file)
parsed_dotenv = dotenv.dotenv_values()
formatted_data_dir = parsed_dotenv["formatted_data_dir"]
model_dir = parsed_dotenv["model_dir"]
model_save_dir = os.path.join(parsed_dotenv["model_dir"], "nathan_tests") # HACK

# ------------------------------

def main():

    ## LOAD DATA

    data_path = os.path.join(formatted_data_dir, 'season10_3D_labeled')
    point_cloud_ids = [os.path.basename(x).split('.')[0] for x in glob.glob(os.path.join(data_path, "*.label"))]
    training_ids, test_val_ids = train_test_split(point_cloud_ids, train_size=0.75)
    testing_ids, validation_ids = train_test_split(test_val_ids, train_size=0.5)


    ## CONFIGURE ML

    cfg_file = os.path.join(model_dir, "configs", "randlanet_season10.yml")
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    cfg.dataset['train_dir'] = data_path
    cfg.dataset['test_dir']  = data_path
    cfg.dataset['val_dir']   = data_path
    cfg.dataset['training_split']   = training_ids
    cfg.dataset['test_split']       = testing_ids
    #cfg.dataset['val_split']       = validation_ids
    cfg.dataset['validation_split'] = validation_ids
    #cfg.dataset['resample_n'] = 5000  # resample each pc to 5000 points

    from supervisely_dataset import Supervisely
    dataset = Supervisely(dataset_path=data_path, **cfg.dataset)

    if(USE_KITTI):
        cfg_file = "/home/equant/.local/lib/python3.8/site-packages/open3d/_ml3d/configs/randlanet_semantickitti.yml"
        cfg = _ml3d.utils.Config.load_from_file(cfg_file)

        dataset = ml3d.datasets.SemanticKITTI(
            dataset_path='/media/equant/7fe7f0a0-e17f-46d2-82d3-e7a8c25200bb/work/SemanticKITTI/',
            test_split=[ '12' ],
            training_split=[ '10' ],
            validation_split=['08'],
            all_split=[ '08', '10', '12' ],
        ) 


    model = ml3d.models.RandLANet()
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, max_epoch=7)

    pipeline.cfg_tb = {
        'readme': "Read me file",
        'cmd_line': "cmd_line",
        'dataset': pprint.pformat(cfg.dataset, indent=2),
        'model': pprint.pformat(cfg.model, indent=2),
        'pipeline': pprint.pformat(pipeline.cfg.cfg_dict, indent=2)
    }


    # print the attributes of the first training pointcloud
    print(dataset.get_split('train').get_attr(0))
    pipeline.run_train()

    # Need to improve this, but for now...
    pipeline.model.save_weights(os.path.join(model_save_dir, "last_model_save_weights.foo"))
    pipeline.model.save(os.path.join(model_save_dir, "last_model_save.foo"))

    #for split_type in ['validation', 'test', 'training']:
    for split_type in ['test', 'training']:
        pcs_with_pred = pred_custom_data(dataset.get_split(split_type), pipeline)
        visualize_pcs_with_pred(pcs_with_pred)

    ## VIEW

def visualize_pcs_with_pred(pcs_with_pred):

    season10_labels = {     # TODO: grab from class.
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

    v.visualize(pcs_with_pred)

def pred_custom_data(data_split, pipeline):
    vis_points = []


    for idx in range(len(data_split)):
        data_attributes = data_split.get_attr(idx)
        name = data_attributes['name']
        data = data_split.get_data(idx)

        results = pipeline.run_inference(data)
        pred_label = (results['predict_labels'] + 1).astype(np.int32)
        # WARNING, THIS IS A HACK
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label[0] = 0

        label = data['label']
        pts = data['point']

        vis_d = {
            "name": name,
            "points": pts,
            "labels": label,
            "pred": pred_label,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_model",
            "points": pts,
            "labels": pred_label,
        }
        vis_points.append(vis_d)

    return vis_points


if __name__ == "__main__":
    main()
