#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
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
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--depth_supervision_lambda", default=0.0, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--save_mesh", action="store_true")
	parser.add_argument("--save_mesh_path", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=512, type=int, help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")

	## na_test
	parser.add_argument('--test_camera_view', type=int,default=0)
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--render_img_HW', type=int, default=None)
	parser.add_argument("--shaded_mesh", action='store_true')
	parser.add_argument("--white_bkgd", action='store_true')

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()

	
	
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


	if args.load_snapshot:
		print("Loading snapshot ", args.load_snapshot)
		testbed.load_snapshot(args.load_snapshot)
	else:
		testbed.reload_network_from_file(network)

	
	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw*sh > 1920*1080*4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh)

	testbed.shall_train = args.train if args.gui else True


	testbed.nerf.render_with_camera_distortion = False
	# testbed.nerf.render_with_camera_distortion = True

	network_stem = os.path.splitext(os.path.basename(network))[0]

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

		# Optionally match nerf paper behaviour and train on a
		# fixed white bg. We prefer training on random BG colors.
		# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
		# testbed.nerf.training.random_bg_color = False

	old_training_step = 0
	n_steps = args.n_steps
	if n_steps < 0 and args.train:
		n_steps = 100000

	
	tqdm_last_update = 0
	if n_steps > 0:
		if args.depth_supervision_lambda>0.0:
			testbed.nerf.training.depth_supervision_lambda=args.depth_supervision_lambda
		print("depth_supervision_lambda:", testbed.nerf.training.depth_supervision_lambda)
		with tqdm(desc="Training", total=n_steps, unit="step") as t:
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break
				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if now - tqdm_last_update > 0.1:
					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now

	if args.save_snapshot:
		print("Saving snapshot ", args.save_snapshot)
		testbed.save_snapshot(args.save_snapshot, False)

	if args.save_mesh_path:
		res = args.marching_cubes_res or 256
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh_path}. Resolution=[{res},{res},{res}]")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh_path, [res, res, res], invert_normals=True)
