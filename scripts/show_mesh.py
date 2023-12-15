import open3d as o3d
import numpy as np
import os
import json
import cv2 as cv


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


def projection(mesh_path, transform_path, out_path):
    transform_json={}
    with open(transform_path, 'r') as f:
        transform_json = json.load(f)
    width = transform_json["frames"][0]["w"]
    height = transform_json["frames"][0]["h"]
    # 投影的图像
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    for frame in transform_json["frames"]:
        file_path = frame["file_path"]
        file_name = file_path[file_path.rfind('/')+1:]
        file_idx = file_name[0:file_name.rfind('.')]
        frame_0_cw_nerf =  np.array(frame["transform_matrix"], dtype=np.float32)
        intrisic = np.array([[frame["fl_x"], 0.0, float(frame["cx"])],
                            [0.0, frame["fl_y"], float(frame["cy"])],
                            [0.0, 0.0, 1.0]
                            ],  dtype=np.float32)
        frame_cw_nerf_34 = frame_0_cw_nerf[0:3, :]
        scale = 1.0
        offset = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        # scale = 0.25
        # offset = np.array([[0.5], [0.5], [0.5]], dtype=np.float32)
        frame_cw_ngp = nerf_matrix_to_ngp(frame_cw_nerf_34, scale, offset)
        trans_c2w_ngp = np.vstack((frame_cw_ngp, np.array([0.0,0.0,0.0,1.0])))
        camera_w2c = np.linalg.inv(trans_c2w_ngp)
        # print(camera_w2c)
        # break
        
        mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(mask.shape)
        for i in range (len(mesh.vertices)):
            vert = np.asarray(mesh.vertices[i], dtype=np.float64)
            # print(vert.shape, vert)
            vert_41 =  np.array([[vert[0]],[vert[1]],[vert[2]],[1.0]])
            vert_cam = np.dot(camera_w2c, vert_41)
            # print("vert_cam:", vert_cam)

            # print("vert_cam[0:3,:]:",vert_cam[0:3,:])
            vert_pixel = np.dot(intrisic, vert_cam[0:3,:])
            # print("vert_pixel:",vert_pixel)

            u = vert_pixel[0,0] / vert_pixel[2,0]
            v = vert_pixel[1,0] / vert_pixel[2,0]

            int_u = int(u+0.5)
            int_v = int(v+0.5)
            if(int_u <0 or int_v<0 or  int_u>= width or int_v >= height):
                continue
            # print("u:",int_u," v:", int_v)

            mask[int_v, int_u] = 255

            # input()
        cv.imwrite(os.path.join(out_path, file_idx+".png"), mask)


def show_point(mesh_path,out_point_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print(mesh_path)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(mesh.vertices)
    o3d.visualization.draw_geometries([mesh])
    # o3d.visualization.draw_geometries([point_cloud])

    # o3d.io.write_point_cloud(out_point_path, point_cloud)



    # triangles = np.asarray(mesh.triangles)
    # normals = np.asarray(mesh.vertex_normals)

    # new_mesh = o3d.geometry.TriangleMesh()
    # new_mesh.vertices = mesh.vertices
    # new_mesh.triangles = mesh.triangles

    # # 计算平均法向并替换三角面的法向
    # triangle_normals = []
    # for i, face in enumerate(triangles):
    #     vertex_indices = np.asarray(face)
    #     face_normals = normals[vertex_indices]
    #     average_normal = np.mean(face_normals, axis=0)
    #     triangle_normals.append(average_normal)

    # new_mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals)

    # # 创建一个渲染窗口并显示
    # o3d.visualization.draw_geometries([mesh])





def test1():
    mesh_path = "/home/fhy/new_disk1/lhw/NeuS2/data/person2/expirement/1102/neus2_mesh_depth00.ply"
    transforms_path = "/home/fhy/new_disk1/lhw/NeuS2/data/person2/render/transforms_render.json"
    out_temp_path = "/home/fhy/new_disk1/lhw/NeuS2/data/person2/render/temp/"
    projection(mesh_path, transforms_path, out_temp_path)
    

def test2():
    root_path = "/home/lhw/Gradute/RenderAndRecon/NeuS2/data/person2/texture"
    mesh_path = os.path.join(root_path, "possion_mesh.ply")
    # mesh_path = os.path.join(root_path, "neus_mesh.ply")
    out_point_path = os.path.join(root_path, "neus_point.ply")
    show_point(mesh_path,out_point_path)


if __name__ == "__main__":
    test1()





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

