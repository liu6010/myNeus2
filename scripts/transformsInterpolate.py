import os
import open3d as o3d
import cv2
import numpy as np
import shutil
import argparse
import json
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; "
                                                 "optionally convert video to images, and optionally run colmap "
                                                 "in the first place")

    parser.add_argument("--transforms_path", default="", help="run ffmpeg first to convert a provided video file into "
                                                       "a set of images. uses the video_fps parameter also")
    parser.add_argument("--out_transforms_path", default="")
    parser.add_argument("--data_name", default="person_gn")

    arguments = parser.parse_args()
    return arguments

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

# https://blog.csdn.net/xwb_12340/article/details/132356418



def interpolate_orientation(start_orientation, end_orientation, degrees = False):

    key_rots = Rotation.from_euler('xyz', [start_orientation, end_orientation], degrees)

    key_times = [0, 2]
    print(key_rots.as_euler('xyz', degrees=True))
    slerp = Slerp(key_times, key_rots)
    times = [0,1,2]
    interp_rots = slerp(times)

    return interp_rots
def interpolate_orientation_matrix(start_orientation, end_orientation, num_points=1):

    key_rots = Rotation.from_matrix([start_orientation, end_orientation])

    key_times = [0, num_points+1]
    # key_times = [0, 2]
    # print(key_rots.as_euler('xyz', degrees=True))
    slerp = Slerp(key_times, key_rots)
    times = list(range(1, num_points+1,1))
    # print("times:", times)
    interp_rots = slerp(times)

    return interp_rots.as_matrix()

def interpolate_position(start_pos, end_pos, num_points=1):
    delta_pos = end_pos - start_pos
    interval = delta_pos / (num_points+1)
    positions = []

    for i in range(1,num_points+1):
        pos = start_pos + i * interval
        positions.append(pos)

    return np.array(positions)

def create_rotation_matrix(axis, angle_degrees):
    # 将角度转换为弧度
    angle_radians = np.radians(angle_degrees)

    # 根据绕定轴旋转的旋转矩阵
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")

    return rotation_matrix



if __name__ == "__main__":
    args = parse_args()
    transforms_path = args.transforms_path
    out_transforms_path = args.out_transforms_path
    STEP=13
    CAMNUM=3

    transform_json={}
    
    with open(transforms_path, 'r') as f:
        transform_json = json.load(f)


    num_len = len(transform_json["frames"])
    map_transform_json = {}
    for frame in transform_json["frames"]:
        file_path = frame["file_path"]
        file_name = file_path[file_path.rfind('/')+1:]
        file_idx = file_name[0:file_name.rfind('.')]
        map_transform_json[int(file_idx)] = np.matrix(frame["transform_matrix"])


    out_transform_json = copy.deepcopy(transform_json)
    # out_transform_json["frames"].clear()
    cnt = -1
    cam_id = 2
    if args.data_name == "person_gn":
        for idx in range(num_len):
            frame = transform_json["frames"][idx]
            file_path = frame["file_path"]
            file_name = file_path[file_path.rfind('/')+1:]
            file_idx = file_name[0:file_name.rfind('.')]

            nerf_c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            nerf_w2c = np.linalg.inv(nerf_c2w)
            virtual_RT_44 = np.eye(4)
            if(file_idx == "0000"):
                print(file_idx)
                cnt+=1
                virtual_R = np.eye(3)
                virtual_t = np.array([0, -1, 0]).reshape(3,1)
                R_temp = copy.deepcopy(nerf_w2c[:3,:3]).reshape(3,3)
                virtual_R = np.linalg.inv(R_temp)
                virtual_R = np.dot(virtual_R, create_rotation_matrix('x', -37))
                virtual_R = np.dot(virtual_R, create_rotation_matrix('y', -8))
                virtual_R = np.dot(virtual_R, create_rotation_matrix('z', -45))

                virtual_RT_44[:3, :3] = virtual_R
                virtual_RT_44[:3, 3] = virtual_t[:3,0]

                nerf_virtual_w2c = np.dot(virtual_RT_44, nerf_w2c)
                nerf_virtual_c2w = np.linalg.inv(nerf_virtual_w2c)

                frame_cp = copy.deepcopy(frame)
                frame_cp["transform_matrix"] = nerf_virtual_c2w.tolist()
                frame_cp["file_path"] = "rgba/%04d.png"%(num_len+cnt)
                out_transform_json["frames"].append(frame_cp)
            
            if(file_idx == "0038"):
                for i in range(5):
                    print(file_idx)
                    cnt+=1
                    virtual_R = np.eye(3)
                    # virtual_t = np.array([0.2*i, 0.0, 0.0]).reshape(3,1)
                    virtual_t = np.array([-0.2*2, 0.3*2, 0]).reshape(3,1)

                    R_temp = copy.deepcopy(nerf_c2w[:3,:3]).reshape(3,3)
                    # virtual_R = np.linalg.inv(R_temp)
                    # virtual_R = np.dot(virtual_R, create_rotation_matrix('x', -15*i))
                    virtual_R = np.dot(create_rotation_matrix('x', 10+i*4), virtual_R )
                    virtual_R = np.dot(create_rotation_matrix('y', 10+i*4), virtual_R )
                    virtual_R = np.dot(create_rotation_matrix('z', 0), virtual_R )
                    # virtual_R = np.dot(virtual_R, create_rotation_matrix('y', i*10))
                    # virtual_R = np.dot(virtual_R, create_rotation_matrix('z', 315))
                    virtual_RT_44[:3, :3] = virtual_R
                    virtual_RT_44[:3, 3] = virtual_t[:3,0]
                    print(nerf_c2w[:3, 3])

                    nerf_virtual_c2w = np.dot(virtual_RT_44, nerf_c2w)
                    # print("nerf_virtual_w2c:", nerf_virtual_w2c)
                    # nerf_virtual_c2w = np.linalg.inv(nerf_virtual_w2c)
                        # print("nerf_virtual_c2w:",nerf_virtual_c2w)

                    frame_cp = copy.deepcopy(frame)
                    frame_cp["transform_matrix"] = nerf_virtual_c2w.tolist()
                    frame_cp["file_path"] = "rgba/%04d.png"%(num_len+cnt)
                    
                    out_transform_json["frames"].append(frame_cp)
                for i in range(5):
                    print(file_idx)
                    cnt+=1
                    virtual_R = np.eye(3)
                    # virtual_t = np.array([0.2*i, 0.0, 0.0]).reshape(3,1)
                    virtual_t = np.array([-0.2*2 +0.2, 0.3*2+0.2, 0]).reshape(3,1)

                    R_temp = copy.deepcopy(nerf_c2w[:3,:3]).reshape(3,3)
                    # virtual_R = np.linalg.inv(R_temp)
                    # virtual_R = np.dot(virtual_R, create_rotation_matrix('x', -15*i))
                    virtual_R = np.dot(create_rotation_matrix('x', 15+4*i), virtual_R )
                    virtual_R = np.dot(create_rotation_matrix('y', 15+4*i), virtual_R )
                    virtual_R = np.dot(create_rotation_matrix('z', 10), virtual_R )
                    # virtual_R = np.dot(virtual_R, create_rotation_matrix('y', i*10))
                    # virtual_R = np.dot(virtual_R, create_rotation_matrix('z', 315))
                    virtual_RT_44[:3, :3] = virtual_R
                    virtual_RT_44[:3, 3] = virtual_t[:3,0]
                    print(nerf_c2w[:3, 3])

                    nerf_virtual_c2w = np.dot(virtual_RT_44, nerf_c2w)
                    # print("nerf_virtual_w2c:", nerf_virtual_w2c)
                    # nerf_virtual_c2w = np.linalg.inv(nerf_virtual_w2c)
                        # print("nerf_virtual_c2w:",nerf_virtual_c2w)

                    frame_cp = copy.deepcopy(frame)
                    frame_cp["transform_matrix"] = nerf_virtual_c2w.tolist()
                    frame_cp["file_path"] = "rgba/%04d.png"%(num_len+cnt)
                    
                    out_transform_json["frames"].append(frame_cp)
        
    for cam_id in range(3):
        startIdx = cam_id*STEP
        endIdx = (cam_id+1)*STEP
        for frame in transform_json["frames"]:
            file_path = frame["file_path"]
            file_name = file_path[file_path.rfind('/')+1:]
            file_idx = file_name[0:file_name.rfind('.')]
            cur_idx = int(file_idx)
            if(cur_idx <startIdx or cur_idx >= endIdx-1):
                continue
            cnt += 1
            
            next_idx = (cur_idx+1) % STEP + startIdx
            cur_matrix_c2w = np.matrix(frame["transform_matrix"])
            next_martrix_c2w = map_transform_json[next_idx]

            cur_nerf_w2c = np.linalg.inv(cur_matrix_c2w)
            next_nerf_w2c = np.linalg.inv(next_martrix_c2w)

            cur_nerf_w2cR = cur_nerf_w2c[:3,:3]
            next_nerf_w2cR = next_nerf_w2c[:3,:3]

            cur_nerf_w2cT = cur_nerf_w2c[:3,3]
            next_nerf_w2cT = next_nerf_w2c[:3,3]

            interRot = interpolate_orientation_matrix(cur_nerf_w2cR, next_nerf_w2cR)
            interTran = interpolate_position(cur_nerf_w2cT, next_nerf_w2cT)

            # print(interRot.shape)
            # print(interTran.shape)

            inter_w2cT = np.append(interRot[0], interTran[0], axis=1)
            inter_w2cT = np.append(inter_w2cT, [[0., 0., 0., 1.0]], axis=0)

            inter_c2wT = np.linalg.inv(inter_w2cT)

            frame_cpy = copy.deepcopy(frame)
            frame_cpy["transform_matrix"] = inter_c2wT.tolist()
            frame_cpy["file_path"] = "rgba/%04d.png"%(num_len+cnt)

            out_transform_json["frames"].append(frame_cpy)
    for cam_id in range(2):
        startIdx = cam_id*STEP
        endIdx = (cam_id+1)*STEP
        for frame in transform_json["frames"]:
            file_path = frame["file_path"]
            file_name = file_path[file_path.rfind('/')+1:]
            file_idx = file_name[0:file_name.rfind('.')]
            cur_idx = int(file_idx)
            if(cur_idx <startIdx or cur_idx >= endIdx):
                continue
            
            
            next_idx = (cur_idx+1) % STEP + endIdx
            cur_matrix_c2w = np.matrix(frame["transform_matrix"])
            next_martrix_c2w = map_transform_json[next_idx]

            cur_nerf_w2c = np.linalg.inv(cur_matrix_c2w)
            next_nerf_w2c = np.linalg.inv(next_martrix_c2w)

            cur_nerf_w2cR = cur_nerf_w2c[:3,:3]
            next_nerf_w2cR = next_nerf_w2c[:3,:3]

            cur_nerf_w2cT = cur_nerf_w2c[:3,3]
            next_nerf_w2cT = next_nerf_w2c[:3,3]

            times = 1
            interRot = interpolate_orientation_matrix(cur_nerf_w2cR, next_nerf_w2cR, times)
            interTran = interpolate_position(cur_nerf_w2cT, next_nerf_w2cT, times)

            # print(interRot.shape)
            # print(interTran.shape)
            for ii_idx in range(times):
                inter_w2cT = np.append(interRot[ii_idx], interTran[ii_idx], axis=1)
                inter_w2cT = np.append(inter_w2cT, [[0., 0., 0., 1.0]], axis=0)

                inter_c2wT = np.linalg.inv(inter_w2cT)
                cnt += 1
                frame_cpy = copy.deepcopy(frame)
                frame_cpy["transform_matrix"] = inter_c2wT.tolist()
                frame_cpy["file_path"] = "rgba/%04d.png"%(num_len+cnt)

                out_transform_json["frames"].append(frame_cpy)

    print(f"writing {out_transforms_path}")
    with open(out_transforms_path, "w") as outfile:
        json.dump(out_transform_json, outfile, indent=2)
