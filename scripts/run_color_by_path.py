#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import commentjson as json

import numpy as np

import sys
import time

from common import *
from render_utils import render_img_training_view

from shutil import copyfile
from tqdm import tqdm


import pyngp as ngp # noqa

from torch.utils.tensorboard import SummaryWriter


def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")
	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")

	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--near_distance", default=-1, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default")

	parser.add_argument("--npg_camera_path", default="", help="Path to a nerf style transforms.json from which to save screenshots.", required=True)
	parser.add_argument("--screenshot_transforms_out", default="", help="Path to a nerf style transforms.json from which to save screenshots.", required=True)
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")
	parser.add_argument("--render_mode", default="", help="Set render mode.")


	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")

	parser.add_argument("--fov", type=float, default=50.0)
	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.", required=True)
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.", required=True)

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")


	args = parser.parse_args()
	return args

def nerf_matrix_to_ngp(nerf_matrix, scale, offset, from_na=False, from_mitsuba=False):
    result = np.copy(nerf_matrix)
    result[:, 1] *= -1
    result[:, 2] *= -1
    result[:, 3] = result[:, 3] * scale + offset[:, 0]

    
    if from_na:
        result[:, 1] *= -1
        result[:, 2] *= -1
    elif from_mitsuba:
        result[:, 0] *= -1
        result[:, 2] *= -1
    else:
        # Cycle axes xyz<-yzx
        tmp = np.copy(result[0, :])
        result[0, :] = np.copy(result[1, :])
        result[1, :] = np.copy(result[2, :])
        result[2, :] = tmp
    
    return result

def ngp_matrix_to_nerf(ngp_matrix, scale, offset, from_na=False, from_mitsuba=False):
    result = np.copy(ngp_matrix)
    
    if from_na:
        result[:, 1] *= -1
        result[:, 2] *= -1
    elif from_mitsuba:
        result[:, 0] *= -1
        result[:, 2] *= -1
    else:
        # 循环轴 xyz->yzx
        tmp = np.copy(result[0, :])
        result[0, :] = np.copy(result[2, :])
        result[2, :] = np.copy(result[1, :])
        result[1, :] = tmp
    
    result[:, 1] *= -1
    result[:, 2] *= -1
    result[:, 3] = (result[:, 3] - offset[:,0]) / scale
    
    return result

def quaternion2rot(quaternion):
	quaternion /= np.linalg.norm(quaternion)
	x, y, z, w = quaternion
	rotation_matrix = np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])
	return rotation_matrix


if __name__ == "__main__":
	args = parse_args()

	# os.makedirs(os.path.join(args.output_path,"checkpoints"), exist_ok=True)
	# os.makedirs(os.path.join(args.output_path,"mesh"), exist_ok=True)
	
	time_name = time.strftime("%m_%d_%H_%M", time.localtime())

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
		with open(args.scene) as f:
			transforms = json.load(f)
		offset = np.array([transforms["offset"]]).reshape(3,1)
		scale = transforms["scale"]
		print("scale:", scale)
		print("offset:", offset)
		# offset = np.array([0.0,0.0,0.0]).reshape(3,1)
		# scale = 1.0
		# print("offset:", offset)

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
	print("args.render_mode:",args.render_mode)

	ref_transforms = {}
	if args.npg_camera_path: # try to load the given file straight away
		print("npg camera path from ", args.npg_camera_path)
		# ref_transforms = {
		# 			"offset":transforms["offset"],
		# 			"scale": transforms["scale"],
		# 			"frames": []
		# 			}
		ref_transforms = {
					"offset":[0.0,0.0,0.0],
					"scale": 1,
					"frames": []
					}
		startIdx = len(transforms["frames"])
		with open(args.npg_camera_path) as f:
			temp_transforms = json.load(f)
		for temp_frame in temp_transforms["path"]:
			R_roate = temp_frame["R"]
			T = np.array([temp_frame["T"]]).reshape(3,1)

			# print("R_roate:", R_roate)
			# print("T:", T)
			R_33 = quaternion2rot(R_roate)
			# print("R_33:",R_33)

			c2w_ngp = np.append(R_33, T, axis=1)
			# print("c2w_ngp:", c2w_ngp)

			c2w_nerf = ngp_matrix_to_nerf(c2w_ngp, scale, offset)
			c2w_nerf = np.append(c2w_nerf, [[0., 0., 0., 1.0]], axis=0)

			file_name = os.path.join(args.screenshot_dir, str(startIdx).zfill(4)+".jpg")

			fov = args.fov
			angle_y = fov * np.pi / 180
			angle_x = angle_y
			# print(fov, angle_x)

			cur_height = args.height
			cur_width = args.width
			fl_y = cur_height / (2*np.tan(angle_y/2.0))
			# print(2*np.tan(angle_y/2.0),fl_y)
			fl_x = cur_width / (2*np.tan(angle_x/2.0))

			# 内参给定fx=fy
			frame = {"file_path":file_name, "transform_matrix":c2w_nerf, "camera_angle_x": angle_x, "camera_angle_y": angle_y, 
					"fl_x": fl_x, "fl_y": fl_y , "cx": int(cur_width/2) , 
					"cy": int(cur_height/2), "w": cur_width, "h": cur_height}
			ref_transforms["frames"].append(frame)
			startIdx += 1

			# print("c2w_ngp:", c2w_ngp)
		for f in ref_transforms["frames"]:
			f["transform_matrix"] = f["transform_matrix"].tolist()
		with open(args.screenshot_transforms_out, "w") as outfile:
			json.dump(ref_transforms, outfile, indent=2)




	testbed.shall_train = args.train if args.gui else True


	testbed.nerf.render_with_camera_distortion = False

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")
		testbed.color_space = ngp.ColorSpace.SRGB
		testbed.nerf.cone_angle_constant = 0


	if ref_transforms:
		# testbed.fov_axis = 0
		screenshot_frames = range(len(ref_transforms["frames"]))
		# print(screenshot_frames)
		
		for idx in screenshot_frames:
			f = ref_transforms["frames"][int(idx)]
			cam_matrix = f["transform_matrix"]
			testbed.fov = f["camera_angle_x"] * 180 / np.pi


			testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])

			outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))

			if not os.path.splitext(outname)[1]:
				if args.render_mode == "depth":
					outname = outname + ".png"
				elif args.render_mode == "shade":
					outname = outname + ".jpg"

			print(f"rendering {outname}")
			render_width = int(f["w"])
			render_height = int(f["h"])
			print("render_width:",render_width, "render_height:",render_height)
			image = testbed.render(render_width, render_height, args.screenshot_spp, True)
			os.makedirs(os.path.dirname(outname), exist_ok=True)
			write_image(outname, image)
