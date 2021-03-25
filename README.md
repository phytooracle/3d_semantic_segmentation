# 3d_semantic_segmentation

## Requirements

### Setup dotenv
```bash
conda install -c conda-forge python-dotenv 
cp sample.env .env
```

## All about our custom implementation of open3d-ml

### When might we get away without using a custom/hacked version?

If the developers solve github issues #241 or #221 it'd may be possible to use the supervisely dataset class (see 3d_semantic_segmentation/src/misc_nathan/supervisely_dataset.py).

This is the current problematic error...

```
AttributeError: 'SemanticSegmentation' object has no attribute 'cfg_tb'
```

### LettuceKITTI

You need to get the lettuce dataset from supervise.ly and then convert it to be formatted as a KITTI dataset.

- download data from supervise.ly
- edit src/data/ConvertSuperviselyToSemanticKitti.py
    - data path
    - output path
    - training split
- run src/data/ConvertSuperviselyToSemanticKitti.py

### Getting python to see our version of open3d-ml

There might be a better way to do this, but here's how I did it...

```
mv /home/equant/.local/lib/python3.8/site-packages/open3d/_ml3d /home/equant/.local/lib/python3.8/site-packages/open3d/_ml3d.original
ln -s /path/to/repo/3d_semantic_segmentation/Open3D-ML/ml3d /home/equant/.local/lib/python3.8/site-packages/open3d/_ml3d
```

### Running Training

```
cd Open3D-ML/scripts
python run_pipeline.py -m RandLANet -p SemanticSegmentation -d SemanticKITTI --dataset_path /media/equant/7fe7f0a0-e17f-46d2-82d3-e7a8c25200bb/work/lettuceKITTI --split train tf
```

### Files within open3d-ml that were edited...

Until we find a way to create our own dataset class, we need to edit things related to the Semantic Kitti dataset within open3d-ml...

| open3dml/Open3D-ML/ml3d/datasets/semantickitti.py             | Removed directories, need to change labels |
|---------------------------------------------------------------|--------------------------------------------|
| open3dml/Open3D-ML/ml3d/datasets/utils/dataprocessing.py      | changed load_label_kitti()                 |
| randlanet_semantic3d.yml                                      | steps_per_epoch_train, max_epoch           |
| semantic_segmentation.yml                                     | max_epoch                                  |
| open3dml/Open3D-ML/ml3d/tf/pipelines/semantic_segmentation.py | max_epoch = 10 in __init__() args          |

### Things we might need to edit

Again, the yaml/yml files hurt my brain, but these files might be important as we move forward...

- Open3D-ML/ml3d/datasets/utils/semantic-kitti.yaml
- Open3D-ML/ml3d/configs/default_cfgs/semantickitti.yml
- Open3D-ML/ml3d/configs/default_cfgs/randlanet.yml
- Open3D-ML/ml3d/configs/randlanet_semantickitti.yml

