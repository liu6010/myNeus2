import os
import open3d as o3d
import cv2
import numpy as np
import shutil
import argparse
import json
import matplotlib.pyplot as plt
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; "
                                                 "optionally convert video to images, and optionally run colmap "
                                                 "in the first place")

    parser.add_argument("--transforms", default="", help="run ffmpeg first to convert a provided video file into "
                                                       "a set of images. uses the video_fps parameter also")
    parser.add_argument("--out_transforms", default="")
    arguments = parser.parse_args()
    return arguments
def quaternion2rot(quaternion):
    quaternion /= np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    rotation_matrix = np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])
    return rotation_matrix
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




def meshNormalReverse(neus_mesh_path, dst_neus_mesh_path):
    mesh = o3d.io.read_triangle_mesh(neus_mesh_path)
    for i in range(len(mesh.vertex_normals)):
        mesh.vertex_normals[i] = -mesh.vertex_normals[i]
    o3d.io.write_triangle_mesh(dst_neus_mesh_path, mesh)


def transformsGenRender(json_path, out_json_path):

    transform_json={}
    with open(json_path, 'r') as f:
        transform_json = json.load(f)

    num_len = len(transform_json["frames"])

    trans_map = {}
    for frame in transform_json["frames"]:
        file_path = frame["file_path"]
        # print(file_path)
        file_name = file_path[file_path.rfind('/')+1:]
        file_idx = file_name[0:file_name.rfind('.')]
        trans_map[int(file_idx)] = np.array(frame["transform_matrix"], dtype=np.float32)
    
    transform_json_new = copy.deepcopy(transform_json)
    transform_json_new["frames"].clear()
    cnt = -1
    for idx in range(num_len):
        frame = transform_json["frames"][idx]
        file_path = frame["file_path"]
        file_name = file_path[file_path.rfind('/')+1:]
        file_idx = file_name[0:file_name.rfind('.')]

        nerf_c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        nerf_w2c = np.linalg.inv(nerf_c2w)
        virtual_RT_44 = np.eye(4)

        if(file_idx == "0018" and False):
            cnt+=1
            nerf_c2w_pre = trans_map[int(file_idx) - 13]
            nerf_w2c_pre = np.linalg.inv(nerf_c2w_pre)
            t_cur = nerf_w2c[:3, 3].reshape(3,1)
            t_pre = nerf_w2c_pre[:3, 3].reshape(3,1)
            t_diff = t_cur - t_pre
            # print(t_pre, t_cur, t_diff)
            t_diff -= np.array([[1.8], [0.0], [-3.8]])
            # virtual_R = np.eye(3)
            virtual_R = create_rotation_matrix('x', -18)
            virtual_R = np.dot(virtual_R, create_rotation_matrix('y', -10))
            virtual_t = np.array([-3, 0, 0]).reshape(3,1)
            virtual_RT_44[:3, :3] = virtual_R
            # virtual_RT_44[:3, 3] = virtual_t[:3,0]
            virtual_RT_44[:3, 3] = t_diff[:3,0]/2.0
            # print("virtual_RT_44:",virtual_RT_44)

            nerf_virtual_w2c = np.dot(virtual_RT_44, nerf_w2c)
            # print("nerf_virtual_w2c:", nerf_virtual_w2c)
            nerf_virtual_c2w = np.linalg.inv(nerf_virtual_w2c)
                # print("nerf_virtual_c2w:",nerf_virtual_c2w)

            frame_cp = copy.deepcopy(frame)
            frame_cp["transform_matrix"] = nerf_virtual_c2w.tolist()
            frame_cp["file_path"] = "rgba/%04d.png"%(num_len+cnt)
            
            transform_json_new["frames"].append(frame_cp)
        if(file_idx == "0009" and False):
            print(file_idx)
            cnt+=1
            # print(t_pre, t_cur, t_diff)
            t_diff = np.array([[1.0], [0.0], [2.0]])
            virtual_R = np.eye(3)
            virtual_R = create_rotation_matrix('y', 15)
            # virtual_R = np.dot(virtual_R, create_rotation_matrix('y', -10))
            virtual_t = np.array([-3, 0, 0]).reshape(3,1)
            virtual_RT_44[:3, :3] = virtual_R
            # virtual_RT_44[:3, 3] = virtual_t[:3,0]
            virtual_RT_44[:3, 3] = t_diff[:3,0]
            # print("virtual_RT_44:",virtual_RT_44)

            nerf_virtual_w2c = np.dot(virtual_RT_44, nerf_w2c)
            # print("nerf_virtual_w2c:", nerf_virtual_w2c)
            nerf_virtual_c2w = np.linalg.inv(nerf_virtual_w2c)
                # print("nerf_virtual_c2w:",nerf_virtual_c2w)

            frame_cp = copy.deepcopy(frame)
            frame_cp["transform_matrix"] = nerf_virtual_c2w.tolist()
            frame_cp["file_path"] = "rgba/%04d.png"%(num_len+cnt)
            
            transform_json_new["frames"].append(frame_cp)
        if(file_idx == "0009"):
            print(file_idx)
            cnt+=1
            virtual_R = np.eye(3)
            virtual_t = np.array([-0.2, -0.4, 2.9]).reshape(3,1)
            R_temp = copy.deepcopy(nerf_w2c[:3,:3]).reshape(3,3)
            virtual_R = np.linalg.inv(R_temp)
            virtual_R = np.dot(virtual_R, create_rotation_matrix('x', -45))
            virtual_R = np.dot(virtual_R, create_rotation_matrix('y', -4))
            virtual_R = np.dot(virtual_R, create_rotation_matrix('z', -90))
            # virtual_R = np.dot(virtual_R, create_rotation_matrix('y', i*10))
            # virtual_R = np.dot(virtual_R, create_rotation_matrix('z', 315))
            virtual_RT_44[:3, :3] = virtual_R
            virtual_RT_44[:3, 3] = virtual_t[:3,0]
            print(nerf_w2c[:3, 3])

            nerf_virtual_w2c = np.dot(virtual_RT_44, nerf_w2c)
            # print("nerf_virtual_w2c:", nerf_virtual_w2c)
            nerf_virtual_c2w = np.linalg.inv(nerf_virtual_w2c)
                # print("nerf_virtual_c2w:",nerf_virtual_c2w)

            frame_cp = copy.deepcopy(frame)
            frame_cp["transform_matrix"] = nerf_virtual_c2w.tolist()
            frame_cp["file_path"] = "rgba/%04d.png"%(num_len+cnt)
            
            transform_json_new["frames"].append(frame_cp)
            for i in range(0):
                print(file_idx)
                cnt+=1
                virtual_R = np.eye(3)
                # virtual_t = np.array([0.2*i, 0.0, 0.0]).reshape(3,1)
                virtual_t = np.array([-0.2, -0.4+0.1*i, 3.0]).reshape(3,1)

                R_temp = copy.deepcopy(nerf_w2c[:3,:3]).reshape(3,3)
                virtual_R = np.linalg.inv(R_temp)
                # virtual_R = np.dot(virtual_R, create_rotation_matrix('x', -15*i))
                virtual_R = np.dot(virtual_R, create_rotation_matrix('x', -45))
                virtual_R = np.dot(virtual_R, create_rotation_matrix('y', 0-2))
                virtual_R = np.dot(virtual_R, create_rotation_matrix('z', -90))
                # virtual_R = np.dot(virtual_R, create_rotation_matrix('y', i*10))
                # virtual_R = np.dot(virtual_R, create_rotation_matrix('z', 315))
                virtual_RT_44[:3, :3] = virtual_R
                virtual_RT_44[:3, 3] = virtual_t[:3,0]
                print(nerf_w2c[:3, 3])

                nerf_virtual_w2c = np.dot(virtual_RT_44, nerf_w2c)
                # print("nerf_virtual_w2c:", nerf_virtual_w2c)
                nerf_virtual_c2w = np.linalg.inv(nerf_virtual_w2c)
                    # print("nerf_virtual_c2w:",nerf_virtual_c2w)

                frame_cp = copy.deepcopy(frame)
                frame_cp["transform_matrix"] = nerf_virtual_c2w.tolist()
                frame_cp["file_path"] = "rgba/%04d.png"%(num_len+cnt)
                
                transform_json_new["frames"].append(frame_cp)
    

    
    print(len(transform_json_new["frames"]))
    with open(out_json_path, "w") as outfile:
        json.dump(transform_json_new, outfile, indent=2)
        


    


if __name__ == "__main__":
    # args = parse_args()
    # transforms_path = args.transforms
    # out_transforms_path = args.out_transforms
    # args = parse_args()
    transforms_path = "/home/fhy/new_disk1/lhw/NeuS2/data/person2/bak/transforms.json"
    out_transforms_path = "/home/fhy/new_disk1/lhw/NeuS2/data/person2/render/transforms_render.json"

 
    transformsGenRender(transforms_path, out_transforms_path)
