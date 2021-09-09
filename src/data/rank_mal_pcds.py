import sys, glob, os
import open3d as o3d
import json
import pandas as pd

import dotenv
env_file = dotenv.find_dotenv()
dotenv.load_dotenv(env_file)
parsed_dotenv = dotenv.dotenv_values()
mal_dir = parsed_dotenv['mal_data_dir']

TESTING = False

color_dictionary = {
        'soil_pixel' : [0.3, 0.3, 0.3],
        'plant_pixel' : [0.2, 1, 0.3],
}

def custom_key_action_without_kb_repeat_delay(pcd, df, pcd_name, save_results_path):
    rotating = True

    vis = o3d.visualization.VisualizerWithKeyCallback()
    original_colors = pcd.colors

    def key_action_callback(vis, action, mods):
        nonlocal rotating
        nonlocal pcd
        nonlocal original_colors
        nonlocal df
        pcd.colors = original_colors
        print(action)
        if action == 1:  # key down
            rotating = True
        elif action == 0:  # key up
            rotating = False
        elif action == 2:  # key repeat
            pass
        return True

    def animation_callback(vis):
        nonlocal rotating
        if rotating:
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            #ctr.rotate(10.0, 10.0)

    def a_key_callback(vis, action, mods):
        pcd.paint_uniform_color([.1,.9,.1])
        df.loc[pcd_name, "MAL Quality"] = 'Good'
        return True

    def x_key_callback(vis, action, mods):
        pcd.paint_uniform_color([.9,.1,.1])
        df.loc[pcd_name, "MAL Quality"] = 'Bad'
        return True

    def z_key_callback(vis, action, mods):
        pcd.paint_uniform_color([.9,.9,.1])
        df.loc[pcd_name, "MAL Quality"] = 'OK'
        return True

    def o_key_callback(vis, action, mods):
        df.to_csv(save_results_path, index_label='pcd')
        vis.close()
        return True

    # key_action_callback will be triggered when there's a keyboard press, release or repeat event
    vis.register_key_action_callback(32, key_action_callback)  # space
    vis.register_key_action_callback(65, a_key_callback)
    vis.register_key_action_callback(90, z_key_callback)
    vis.register_key_action_callback(79, o_key_callback)
    vis.register_key_action_callback(88, x_key_callback)

    # animation_callback is always repeatedly called by the visualizer
    vis.register_animation_callback(animation_callback)

    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()


pcd_dir = os.path.join(mal_dir, "ds0", "pointcloud")
ann_dir = os.path.join(mal_dir, "ds0", "ann")

pcd_files = glob.glob(os.path.join(pcd_dir, "*.pcd"))
if TESTING == True:
    pcd_files = pcd_files[0:3]

mal_quality_csv_path = os.path.join(mal_dir, "mal_quality.csv")
if os.path.isfile(mal_quality_csv_path):
    df = pd.read_csv(mal_quality_csv_path, index_col='pcd');
else:
    df = pd.DataFrame([], index=[os.path.basename(x) for x in pcd_files], columns=['MAL Quality'])

with open(os.path.join(mal_dir,'meta.json')) as f:
      meta_data = json.load(f)
with open(os.path.join(mal_dir, 'key_id_map.json')) as f:
      key_id_map_data = json.load(f)

progress_count = 0
for pcd_path in pcd_files:
    progress_count = progress_count + 1
    pcd_name = os.path.basename(pcd_path);
    print(f"{progress_count}/{len(pcd_files)} : {pcd_name}")
    if df.loc[pcd_name, 'MAL Quality'] in ['Good', 'Bad', 'OK']:
        print(f"Skipping pcd: {pcd_name} because it's already been ranked")
        continue
    pcd = o3d.io.read_point_cloud(pcd_path);
    if not pcd.has_points():
        print("Something wrong.  No points in PCD")
        sys.exit(0);

    pcd.paint_uniform_color(color_dictionary['soil_pixel'])
    if not pcd.has_colors():
        print("Something wrong.  No colors in PCD after trying to paint_uniform_color()");
        sys.exit(0);

    with open(os.path.join(ann_dir, f'{pcd_name}.json')) as f:
          ann_data = json.load(f)

    # ['figures'][1]['geometry']['indices'][-1]o

    label_key_to_index_dict = key_id_map_data['objects'] # e.g. plant = 0, soil = 1 or whatever
    labels = dict()

    for obj in ann_data['objects']:
        labels[obj['key']] = obj['classTitle']

    for figure in ann_data['figures']:
        current_key   = figure['objectKey']
        current_label = labels[current_key]
        print(current_label)
        if current_label == 'plant_pixel':
            print(f"Found {len(figure['geometry']['indices'])} pixels to color")
            for _i in figure['geometry']['indices']:
                pcd.colors[_i] = color_dictionary[current_label]

    dpcd = pcd.voxel_down_sample(voxel_size=0.0051)

    custom_key_action_without_kb_repeat_delay(dpcd, df, pcd_name, mal_quality_csv_path)
                                              



#vis = o3d.visualization.Visualizer()
#vis.create_window()
#vis.get_render_option().point_size = 3.0
#vis.add_geometry(dpcd)
#vis.capture_screen_image("file.jpg", do_render=True)
#vis.destroy_window()
