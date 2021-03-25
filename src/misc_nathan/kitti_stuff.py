import open3d.ml as _ml3d
import open3d.ml.torch as ml3d # or open3d.ml.tf as ml3d

dataset = ml3d.datasets.SemanticKITTI(dataset_path='/media/equant/7fe7f0a0-e17f-46d2-82d3-e7a8c25200bb/work/SemanticKITTI/',
    test_split=[ '12' ],
    training_split=[ '10' ],
    validation_split=['08'],
    all_split=[ '08', '10', '12' ],
)

# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('all')

# print the attributes of the first datum
print(all_split.get_attr(0))

# print the shape of the first point cloud
print(all_split.get_data(0)['point'].shape)

# show the first 100 frames using the visualizer
vis = ml3d.vis.Visualizer()
vis.visualize_dataset(dataset, 'all', indices=range(1))


framework = "torch" # or tf
cfg_file = "/home/equant/.local/lib/python3.8/site-packages/open3d/_ml3d/configs/randlanet_semantickitti.yml"

cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.RandLANet(**cfg.model)
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "randlanet_semantickitti_202009090354utc.pth"
randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202009090354utc.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
    os.system(cmd)

pipeline.load_ckpt(ckpt_path=ckpt_path)

test_split = dataset.get_split("test")
data = test_split.get_data(0)

result = pipeline.run_inference(data)

pipeline.run_test()

