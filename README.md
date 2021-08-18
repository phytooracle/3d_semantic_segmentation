# 3d_semantic_segmentation

## Requirements and Installation

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

pip install python-dotenv 

# Edit .env so it has the correct project_dir
cp sample.env .env
vim .env
```



## Data

### ConvertSuperviselyToGeneric 

```
conda activate open3d-ml
cd directory_with_uncompressed_supervisely_data_directory
python ~/work/repos/3d_semantic_segmentation/src/data/ConvertSuperviselyToGeneric.py
python /home/equant/work/repos/3d_semantic_segmentation/src/data/ConvertSuperviselyToGeneric.py -i full_auto_labeled -o full_auto_labeled-generic
```


## Training

```
conda activate open3d-ml
cd src/models/
python train_pred_vis.py
```

An example of how to watch training in progress...
```
cd train_log
cd 00028_RandLANet_SemanticKITTI_tf
tensorboard --logdir=.
firefox http://localhost:6006/
```

### HPC Training

#### put data on xdisk
```
iget -PKVT /iplant/home/travis_simmons/labeled_pcds.tar.gz
tar -zxvf labeled_pcds.tar.gz
```

Login to a Puma Interactive node

Run the training script
```
singularity build 3d_semantic_training.simg docker://phytooracle/3d_semantic_segmentation
singularity run 3d_semantic_training.simg {directory containing your bin / label files} -m {directory containing your model file} -outdir {where you want the train_pred_vis outputs to land}
```

If you change the gihub repository, it will trigger an automatic build on dockerhub. It usually takes about ~20 mins to build.

You can then rm 3d_semantic_training.simg and run the two lines above again.

In order to avoid the 20 min wait time, it is often the best practice to install singularity on your local machine and ...

```
git clone https://github.com/phytooracle/3d_semantic_segmentation.git
cd 3d_semantic_segmentation
singularity build 3d_semantic_training.simg .
singularity run 3d_semantic_training.simg {directory containing your bin / label files} -m {directory containing your model file} -outdir {where you want the train_pred_vis outputs to land}
```
Did not preform how you expected?
```
rm 3d_semantic_training.simg
make changes to repo (don't commit)
cd 3d_semantic_segmentation
singularity build 3d_semantic_training.simg .
singularity run 3d_semantic_training.simg {directory containing your bin / label files} -m {directory containing your model file} -outdir {where you want the train_pred_vis outputs to land}
```
Repeat till you are happy with the results, then commit changes.




