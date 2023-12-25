import open3d as o3d
import numpy as np
import os
import json
import cv2 as cv
import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; "
                                                 "optionally convert video to images, and optionally run colmap "
                                                 "in the first place")

    parser.add_argument("--mesh_path", default="", required=True)
    parser.add_argument("--out_pcl_path", default="", required=True)

    arguments = parser.parse_args()
    return arguments


def export_point(mesh_path,out_point_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print(mesh_path)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(mesh.vertices)
    point_cloud.colors = o3d.utility.Vector3dVector(mesh.vertex_colors)
    point_cloud.normals = o3d.utility.Vector3dVector(mesh.vertex_normals)
    # o3d.visualization.draw_geometries([mesh])
    # o3d.visualization.draw_geometries([point_cloud])

    o3d.io.write_point_cloud(out_point_path, point_cloud, write_ascii=True)




if __name__ == "__main__":
    args = parse_args()
    mesh_path = args.mesh_path
    out_point_path = args.out_pcl_path
    export_point(mesh_path,out_point_path)





# 读取OBJ文件或其他格式的网格
# mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True, print_progress=True)
# mesh = o3d.io.read_triangle_model(mesh_path)

# print("has triangle normals?", mesh.has_vertex_normals())

# print(len(mesh.vertex_normals))
# # 取反法向矢量
# # mesh.compute_vertex_normals()  # 计算法向矢量
# for i in range(len(mesh.vertex_normals)):
#     normal = mesh.vertex_normals[i]
#     mesh.vertex_normals[i] = -normal  # 取反法向矢量
# # mesh.compute_triangle_normals()  # 计算法向矢量

# 显示网格
# o3d.visualization.draw_geometries([mesh])
# out_mesh_path = "/home/lhw/Gradute/datasets/0616UE4/person0616/output/open3d/mesh_01.ply"
# o3d.io.write_triangle_mesh(out_mesh_path, mesh, write_ascii=True)

