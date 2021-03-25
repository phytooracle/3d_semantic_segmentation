import open3d.ml as _ml3d
import open3d.ml.tf as ml3d  # or open3d.ml.tf as ml3d
import supervisely_dataset

dataset_path = '/home/equant/work/sandbox/3D/open3dml/lettuce_semantic_segmentation/converted_to_o3d_dataset'

dataset = supervisely_dataset.Supervisely(dataset_path=dataset_path, train_dir='train', test_dir='test', val_dir='val')
#dataset.get_split('train')



##################################################
#             Semantic Segmentation              #
##################################################

##################################################
#   Training a model for semantic segmentation   #
##################################################

model = ml3d.models.RandLANet()

pipeline = ml3d.pipelines.SemanticSegmentation(model=model, dataset=dataset, max_epoch=100)

# prints training progress in the console.
pipeline.run_train()


