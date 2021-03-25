import open3d.ml as _ml3d
#import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d
import open3d.ml.tf as ml3d  # or open3d.ml.tf as ml3d
import supervisely_dataset

dataset_path = '/home/equant/work/sandbox/3D/open3dml/lettuce_semantic_segmentation/converted_to_o3d_dataset'

dataset = supervisely_dataset.Supervisely(dataset_path=dataset_path, train_dir='train', test_dir='test', val_dir='val')
#dataset.get_split('train')



# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('all')

# print the attributes of the first datum
print(all_split.get_attr(0))

# print the shape of the first point cloud
print(all_split.get_data(0)['point'].shape)



split = dataset.get_split('train')
vis = ml3d.vis.Visualizer()
vis.visualize_dataset(dataset, 'all', indices=range(1))


# show the first pc using the visualizer
for i in range(len(all_split)):
    vis = ml3d.vis.Visualizer()
    vis.visualize_dataset(dataset, 'all', indices=[i])




##################################################
#             Semantic Segmentation              #
##################################################

##################################################
#   Training a model for semantic segmentation   #
##################################################
if True:
    model = ml3d.models.RandLANet()

    pipeline = ml3d.pipelines.SemanticSegmentation(model=model, dataset=dataset, max_epoch=10)

    # prints training progress in the console.
    pipeline.run_train()

##################################################
#    Training a model for 3D object detection    #
##################################################
if False:
    # create the model with random initialization.
    model = ml3d.models.PointPillars(voxel_size=[1,1,1])

    pipeline =  ml3d.pipelines.ObjectDetection(model=model, dataset=dataset, max_epoch=100)

    # prints training progress in the console.
    pipeline.run_train()
