# 3d_semantic_segmentation

## Requirements

```bash
conda create --name open3d-ml python=3.6.9
conda activate open3d-ml
pip install --upgrade pip
pip install open3d==0.11.2
git clone https://github.com/intel-isl/Open3D-ML.git
cd Open3D-ML

pip install -r requirements-tensorflow.txt
# Test it...
python -c "import open3d.ml.tf as ml3d"


cd ~/wherever_you_want_repo_to_live
git clone https://github.com/phytooracle/3d_semantic_segmentation
cd 3d_semantic_segmentation

# Dotenv stuff
pip install python-dotenv 
cp sample.env .env
# Edit .env so it has the correct project_dir
vim .env
```



## Data

First you need to DL the labeled data from supervise.ly

If you want, you can use this sample data of ~17 labeled lettuces" (https://de.cyverse.org/dl/d/C11B162C-DB18-4BA2-8E86-9B2F2FF03CF7/lettuce_semantic_segmentation.tar.gz)

```bash
mkdir -p data/raw/lettuce_semantic_segmentation
cp lettuce_semantic_segmentation.tar.gz data/raw/lettuce_semantic_segmentation
tar -zxvf lettuce_semantic_segmentation.tar.gz
python src/data/ConvertSuperviselyToGeneric.py
```

Now you should have a directory `data/formatted/season10_3D_labeled` with `*.bin` and `*.label` files in it.


## Scripts

This is where the bulk of our to-do is.  See `src/models/train_pred_vis.py`

## Training

At the moment, training happens in `src/models/train_pred_vis.py`.

An example of how to watch training in progress...
```
cd train_log
cd 00028_RandLANet_SemanticKITTI_tf
tensorboard --logdir=.
firefox http://localhost:6006/
```

