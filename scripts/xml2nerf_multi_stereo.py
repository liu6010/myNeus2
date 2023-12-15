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

	parser.add_argument("--xml_in", default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
	parser.add_argument("--n", type=int, default="transforms.json", help="output path")
	parser.add_argument("--w", type=int, default=3840, help="width")
	parser.add_argument("--h", type=int, default=2160, help="height")
	parser.add_argument("--scale", type=float, default=0.5, help="height")
	parser.add_argument("--offset", type=float, default=0.5, help="height")
	parser.add_argument("--out", default="transforms.json", help="output path")
	parser.add_argument("--images", default="images", help="input path to the images")
	parser.add_argument("--aabb_scale", type=int, default=16, choices=[1,2,4,8,16], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
	parser.add_argument("--use_depth", type=lambda x: x.lower() == 'true', required=False)

	parser.add_argument("--depths", default="depths")
	parser.add_argument("--integer_depth_scale", type=float, default=0.001)
	args = parser.parse_args()
	return args

def variance_of_laplacian(image):
    	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

if __name__ == "__main__":
	args = parse_args()
	
	XML_FOLDER = args.xml_in
	OUT_PATH = args.out
	AABB_SCALE = args.aabb_scale
	use_depth = args.use_depth
	width = args.w
	height = args.h
	SCALE = args.scale
	OFFSET = args.offset
	integer_depth_scale = args.integer_depth_scale
	DEPTH_FOLDER = args.depths
	print(f"outputting to {OUT_PATH}...")
	# out = {
	# 	"aabb_scale": AABB_SCALE,
	# 	"frames": [],
	# }
	
	if use_depth:
		out = {
			"w": width,
			"h":height,
			"scale":SCALE,
			"offset":[OFFSET,OFFSET,OFFSET],
			"aabb_scale": AABB_SCALE,
            "enable_depth_loading":True,
            "integer_depth_scale":integer_depth_scale,
            "frames": [],
		}
	else:
		out = {
			"w": width,
			"h":height,
			"scale":SCALE,
			"offset":[OFFSET,OFFSET,OFFSET],
            "aabb_scale": AABB_SCALE,
            "frames": [],
		}
	
	up = np.zeros(3)
	for i in range(args.n):
		name = args.images%i
		if use_depth:
			depth_name = args.depths%i
		b = sharpness(name)
		print(XML_FOLDER%(i))
		cv_file = cv2.FileStorage(XML_FOLDER%(i), cv2.FILE_STORAGE_READ)
		#6DOF
		# m = cv_file.getNode("RT").mat().astype(np.float64)
		# Intrinsic = cv_file.getNode("IntrinsicCam").mat().astype(np.float64)
		# Distortion = cv_file.getNode("DistortionCam").mat().astype(np.float64)
		#fangzhen
		
		# zxs gai!!!!!
		m = cv_file.getNode("ExtrinsicIr2World_new").mat().astype(np.float64)
		#Intrinsic = cv_file.getNode("intrinsicMat").mat().astype(np.float64)
		#R = cv_file.getNode("transR").mat().astype(np.float64)
		#T = cv_file.getNode("transT").mat().astype(np.float64)
		#m = np.append(R, T, axis=1)
		#m = np.append(m, [[0., 0., 0., 1.0]], axis=0)
		Intrinsic = cv_file.getNode("IntrinsicCam").mat().astype(np.float64)
		Distortion = cv_file.getNode("DistortionCam").mat().astype(np.float64)
		Distortion = np.array([[0.0,0.0,0.0,0.0,0.0]])

		# print(m)
		# c2w = np.linalg.inv(m)
		c2w = m
		c2w[0:3,2] *= -1 # flip the y and z axis
		c2w[0:3,1] *= -1
		c2w = c2w[[1,0,2,3],:] # swap y and z
		c2w[2,:] *= -1 # flip whole world upside down

		up += c2w[0:3,1]
		# print(c2w)
		
		fl_x = Intrinsic[0,0]
		fl_y = Intrinsic[1,1]
		cx = Intrinsic[0,2]
		cy = Intrinsic[1,2]
		w = width
		h = height
		angle_x = math.atan(w / (fl_x * 2)) * 2
		angle_y = math.atan(h / (fl_y * 2)) * 2
		fovx = angle_x * 180 / math.pi
		fovy = angle_y * 180 / math.pi
		# print(Intrinsic)
		# print(Distortion)
		# print(c2w.type)
		if use_depth:
			frame={"file_path":name,"depth_path": depth_name, "sharpness":b,"camera_angle_x": angle_x,
				"camera_angle_y": angle_y,"transform_matrix": c2w, "fl_x": fl_x, "fl_y": fl_y , "cx": cx , "cy": cy, "w": w, "h": h, "k1": Distortion[0,0], "k2": Distortion[0,1],"p1": Distortion[0,2], "p2": Distortion[0,3]}
		else:
			frame={"file_path":name, "sharpness":b,"camera_angle_x": angle_x,
			"camera_angle_y": angle_y,"transform_matrix": c2w, "fl_x": fl_x, "fl_y": fl_y , "cx": cx , "cy": cy, "w": w, "h": h, "k1": Distortion[0,0], "k2": Distortion[0,1],"p1": Distortion[0,2], "p2": Distortion[0,3]}
		
		out["frames"].append(frame)	
	print(out)
	nframes = len(out["frames"])
	up = up / np.linalg.norm(up)
	print("up vector was", up)
	R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
	R = np.pad(R,[0,1])
	R[-1, -1] = 1

	for f in out["frames"]:
		f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

	# find a central point they are all looking at
	print("computing center of attention...")
	totw = 0.0
	totp = np.array([0.0, 0.0, 0.0])
	for f in out["frames"]:
		mf = f["transform_matrix"][0:3,:]
		for g in out["frames"]:
			mg = g["transform_matrix"][0:3,:]
			p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
			print(p,w)
			if w > 0.01:
				totp += p*w
				totw += w
	totp /= totw
	print(totp) # the cameras are looking at totp
	for f in out["frames"]:
		f["transform_matrix"][0:3,3] -= totp

	avglen = 0.
	for f in out["frames"]:
		avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
	avglen /= nframes
	print("avg camera distance from origin", avglen)
	for f in out["frames"]:
		f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

	for f in out["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()
	print(nframes,"frames")
	print(f"writing {OUT_PATH}")
	with open(OUT_PATH, "w") as outfile:
		json.dump(out, outfile, indent=2)
