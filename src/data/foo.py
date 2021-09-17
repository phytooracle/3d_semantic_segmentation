import sys, glob, os
import open3d as o3d
import json
import numpy as np
import pandas as pd
import open3d.visualization.gui as gui

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

    def abort(vis, action, mods):
        print("ABORT!")
        sys.exit(0)

    # key_action_callback will be triggered when there's a keyboard press, release or repeat event
    vis.register_key_action_callback(32, key_action_callback)  # space
    #vis.register_key_action_callback(65, a_key_callback)
    vis.register_key_action_callback(ord("A"), a_key_callback)
    vis.register_key_action_callback(90, z_key_callback)
    vis.register_key_action_callback(79, o_key_callback)
    vis.register_key_action_callback(88, x_key_callback)
    vis.register_key_action_callback(88, x_key_callback)
    vis.register_key_action_callback(ord("C"), abort)
    vis.register_key_action_callback(ord("Q"), abort)
    vis.register_key_action_callback(ord("1"), abort)

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
                                              



if False:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 3.0
    vis.add_geometry(dpcd)
    vis.capture_screen_image("file.jpg", do_render=True)
    vis.destroy_window()

if False:


	def on_mouse(e):
		if e.type == gui.MouseEvent.Type.BUTTON_DOWN:
			print("[debug] mouse:", (e.x, e.y))
		return gui.Widget.EventCallbackResult.IGNORED

	def on_key(e):
		if e.key == gui.KeyName.SPACE:
			if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
				print("[debug] SPACE released")
			else:
				print("[debug] SPACE pressed")
			return gui.Widget.EventCallbackResult.HANDLED
		if e.key == gui.KeyName.W:  # eats W, which is forward in fly mode
			print("[debug] Eating W")
			return gui.Widget.EventCallbackResult.CONSUMED
		return gui.Widget.EventCallbackResult.IGNORED

    def checkbox_callback(is_checked):
        print("CHECKBOX YO!")

	def main():
		gui.Application.instance.initialize()
		w = gui.Application.instance.create_window("My Thing",
												   640, 480)
		scene = gui.SceneWidget()
		scene.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
		w.add_child(scene)

        em = w.theme.font_size

        #layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                                     #0.5 * em))

        #collapse = gui.CollapsableVert("Widgets", 0.33 * em,
                                       #gui.Margins(em, 0, 0, 0))
        #_label = gui.Label("Lorem ipsum dolor")
        #gui.Color(1.0, 0.5, 0.0)
        #collapse.add_child(_label)

        #mylabel=gui.Label("FOOOOOOO!")
        #w.add_child(mylabel, visible=True)

        # Create a checkbox. Checking or unchecking would usually be used to set
        # a binary property, but in this case it will show a simple message box,
        # which illustrates how to create simple dialogs.
        #cb = gui.Checkbox("Enable some really cool effect")
        #cb.set_on_checked(checkbox_callback)  # set the callback function
        #collapse.add_child(cb)
        #layout.add_child(collapse)
#
#.add_child(gui.Label("Open3D GUI Example"))
		obj = o3d.geometry.TriangleMesh.create_moebius()
		obj.compute_vertex_normals()
        #foo = makemesh(pcd, max_triangles=100000)
        #foo.compute_vertex_normals()

		material = o3d.visualization.rendering.Material()
		material.shader = "defaultLit"
		scene.scene.add_geometry("Moebius", obj, material)
		#scene.scene.add_geometry("Foo", dpcd, material)
		#scene.scene.add_geometry("Foo", foo, material)
        #w.add_child(gui.Label("Open3D GUI Example"))

		scene.setup_camera(60, scene.scene.bounding_box, (0, 0, 0))
		# scene.set_view_controls(gui.SceneWidget.Controls.FLY)
		scene.set_on_mouse(on_mouse)
		scene.set_on_key(on_key)

		gui.Application.instance.run()



def makemesh(pcd, max_triangles=2000):

    pcd.estimate_normals()
	distances = pcd.compute_nearest_neighbor_distance()
	avg_dist = np.mean(distances)
	radius = 3 * avg_dist
	bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    if len(bpa_mesh.triangle_normals) > max_triangles:
        bpa_mesh = bpa_mesh.simplify_quadric_decimation(max_triangles)
	bpa_mesh.remove_degenerate_triangles()
	bpa_mesh.remove_duplicated_triangles()
	bpa_mesh.remove_duplicated_vertices()
	bpa_mesh.remove_non_manifold_edges()
	return bpa_mesh

	
