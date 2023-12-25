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
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

def parse_args():
	parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

	parser.add_argument("--transforms", default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
	parser.add_argument("--gaussian", default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
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

if __name__ == "__main__":
	args = parse_args()
	
	transforms_path = args.transforms
	gaussian_path = args.gaussian
	with open(transforms_path, 'r') as f:
		transforms = json.load(f)
	scale = 1.0
	offset = np.array([0,0,0]).reshape(3,1)
	for f in transforms["frames"]:
		c2w = np.matrix(f["transform_matrix"])[:3, :]
		ngp_c2w = nerf_matrix_to_ngp(c2w, scale, offset)
		ngp_c2w = np.vstack((ngp_c2w, np.array([0.0,0.0,0.0,1.0])))
		f["transform_matrix"] = ngp_c2w
		file_path = f["file_path"]
		file_name = file_path[file_path.rfind('/')+1:]
		f["file_path"] = "./rgba/"+file_name

	for f in transforms["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()
	print(f"writing {gaussian_path}")
	with open(gaussian_path, "w") as outfile:
		json.dump(transforms, outfile, indent=2)
