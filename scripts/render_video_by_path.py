#!/bin/env python
import argparse
import os
import commentjson as json
import numpy as np
import sys
import time
from common import *
# from scenes import scenes_nerf, scenes_image, scenes_sdf, scenes_volume, setup_colored_sdf
from tqdm import tqdm
import pyngp as ngp # noqa
import open3d as o3d
import time

def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")
	parser.add_argument("--n_seconds", type=int, default=5, help="Set n_seconds.")
	parser.add_argument("--fps", type=int, default=60, help="Set fps.")
	parser.add_argument("--camera_path", default="", help="Set camera path.")
	parser.add_argument("--render_mode", default="", help="Set render mode.")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=1, help="Number of samples per pixel in screenshots.")

	args = parser.parse_args()
	return args

def WriteNumpy(path,m):
	#with open(path,'w') as f:
		
	image = np.array(m)
	image *= 1000
	height = np.size(image, 0)
	width = np.size(image, 1)
	image = np.array(image, dtype=np.int16)
	o3d_depth = o3d.geometry.Image(image)
	o3d.io.write_image(path ,o3d_depth)
		
		
		
		#np.savetxt(f,m,delimiter=' ',newline='\n',header='',footer='',comments='# ')

if __name__ == "__main__":
	args = parse_args()

	mode = ngp.TestbedMode.Nerf 
	configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")

	base_network = os.path.join(configs_dir, "base.json")
	network = args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)


	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(args.sharpen)

	if mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	if args.scene:
		scene = args.scene
		testbed.load_training_data(scene)


	if args.load_snapshot:
		print("Loading snapshot ", args.load_snapshot)
		testbed.load_snapshot(args.load_snapshot)
	else:
		testbed.reload_network_from_file(network)

	if args.render_mode == "ao":
		testbed.render_mode = ngp.RenderMode.AO
	elif args.render_mode == "shade":
		testbed.render_mode = ngp.RenderMode.Shade
	elif args.render_mode == "normals":
		testbed.render_mode = ngp.RenderMode.Normals
	elif args.render_mode == "positions":
		testbed.render_mode = ngp.RenderMode.Positions
	elif args.render_mode == "depth":
		testbed.render_mode = ngp.RenderMode.Depth
	elif args.render_mode == "distance":
		testbed.render_mode = ngp.RenderMode.Distance
	elif args.render_mode == "stepsize":
		testbed.render_mode = ngp.RenderMode.Stepsize
	elif args.render_mode == "distortion":
		testbed.render_mode = ngp.RenderMode.Distortion
	elif args.render_mode == "cost":
		testbed.render_mode = ngp.RenderMode.Cost
	elif args.render_mode == "slice":
		testbed.render_mode = ngp.RenderMode.Slice


	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	if ref_transforms:
		
		
		if not args.screenshot_frames:
			args.screenshot_frames = range(len(ref_transforms["frames"]))
		print(args.screenshot_frames)
		testbed.nerf.rendering_min_transmittance = 1e-4
		testbed.snap_to_pixel_centers = True
		#start_total_time = time.time()
		testbed.fov_axis = 0
		total_time = 0
		for idx in args.screenshot_frames:
			
			f = ref_transforms["frames"][int(idx)]
			file_path = f["file_path"]
			file_name = file_path[file_path.rfind('/')+1:]
			file_idx = file_name[0:file_name.rfind('.')]
			if int(file_idx) <=38:
				continue


			# testbed.reset_camera()
			testbed.fov = f["camera_angle_x"] * 180 / np.pi
			# testbed.fov_xy = [f["camera_angle_x"] * 180 / np.pi, f["camera_angle_y"] * 180 / np.pi]
			# testbed.fov = f["fov"] * 180 / np.pi
			testbed.screen_center = [1.0-f["cx"]/f["w"], 1.0-f["cy"]/f["h"]]
			
			spp = 8
			cam_matrix = f["transform_matrix"]
			testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
			outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))
			
			# Some NeRF datasets lack the .png suffix in the dataset metadata
			if not os.path.splitext(outname)[1]:
				outname = outname + ".jpg"

			outname = outname[: -4] + '.jpg'
			
			
			# print()
			start_get_color_time = time.time()
			image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), spp, True)
			end_get_color_time = time.time()
			total_time += ((end_get_color_time - start_get_color_time))
			print("get id %d"%idx+" time: %fs"%(end_get_color_time - start_get_color_time)) 
			os.makedirs(os.path.dirname(outname), exist_ok=True)
			
			if args.render_mode == "depth":
				txt_fmt = outname[: -4] + '.png'
				# out_txt = args.screenshot_dir + txt_fmt
				print(f"rendering {txt_fmt}")
				WriteNumpy(txt_fmt,image[:,:,0])
			else:
				print(f"rendering {outname}")
				start_write_image_time = time.time()
				write_image(outname, image)
				end_write_image_time = time.time()
				total_time += (end_write_image_time - start_write_image_time)
				print("write id %d"%idx+" time: %fs"%(end_write_image_time - start_write_image_time))
		# end_total_time = time.time()
		print("total time: %fs"%total_time)
			
