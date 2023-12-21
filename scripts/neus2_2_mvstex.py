import os
import open3d as o3d
import cv2
import numpy as np
import shutil
import argparse
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; "
                                                 "optionally convert video to images, and optionally run colmap "
                                                 "in the first place")

    parser.add_argument("--root_path", default="", help="run ffmpeg first to convert a provided video file into "
                                                       "a set of images. uses the video_fps parameter also")
    parser.add_argument("--neus_mesh_path", default="")
    parser.add_argument("--neus_mesh_name", default="")
    parser.add_argument("--render_dir", default="")
    parser.add_argument("--width", default=3840)
    parser.add_argument("--height", default=2160)
    parser.add_argument("--flag", default=0)

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


def custom_draw_geometry_with_camera_trajectory(pcd, render_option_path,
                                                camera_trajectory_path,test_data_path, width, height):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
        o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    custom_draw_geometry_with_camera_trajectory.vis.create_window(width=2160, height=3840)
    image_path = os.path.join(test_data_path, 'image')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    depth_path = os.path.join(test_data_path, 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            # print("Capture image {:05d}".format(glb.index))
            # depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            # plt.imsave(os.path.join(depth_path, '{:05d}.png'.format(glb.index)),
            #            np.asarray(depth),
            #            dpi=1)
            plt.imsave(os.path.join(image_path, '{:05d}.png'.format(glb.index)),
                       np.asarray(image),
                       dpi=1)
            # vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            # vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            # print(glb.trajectory.parameters[glb.index].intrinsic.intrinsic_matrix)
            # print(glb.trajectory.parameters[glb.index].intrinsic.width, " ,", glb.trajectory.parameters[glb.index].intrinsic.height)
            # if((float(glb.trajectory.parameters[glb.index].intrinsic.width)/2.0-0.5) == glb.trajectory.parameters[glb.index].intrinsic.intrinsic_matrix[0,2]):
            #     print("+++++++++++++++++++++++++++")
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index], allow_arbitrary=True)
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    # vis.get_render_option().load_from_json(render_option_path)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


def Pose2Open3d(json_output, json_path):

    transform_json={}
    with open(json_path, 'r') as f:
        transform_json = json.load(f)
        # print(transform_json)
    scale = 1.0
    # if "scale" in transform_json:
    #     scale = transform_json["scale"]
    #     print(scale)
    offset = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)

    # if "offset" in transform_json:
    #     offset = np.array(transform_json["offset"]).reshape(-1, 1)
    #     print(offset.shape, offset)
    paras_dict = {} # 所有相机参数
    int_dict = {}   # 单个相机的内参数据
    paras_list = []
    for frame in transform_json["frames"]:
        file_path = frame["file_path"]
        file_name = file_path[file_path.rfind('/')+1:]
        file_idx = file_name[0:file_name.rfind('.')]
        # print("file idx", file_idx)

        # intrisic = np.array([[frame["fl_x"], 0.0, float(frame["cx"]) / frame["w"]],
        #                       [0.0, frame["fl_y"], float(frame["cy"])/ frame["h"]],
        #                       [0.0, 0.0, 1.0]
        #                       ],  dtype=np.float32)
        intrisic = np.array([[frame["fl_x"], 0.0, float(frame["cx"])],
                              [0.0, frame["fl_y"], float(frame["cy"])],
                              [0.0, 0.0, 1.0]
                              ],  dtype=np.float32)
        # print("intrisic", intrisic)
        # input()
        width = int(frame["w"])
        height = int(frame["h"])
        distortion = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        transform_matrix_c2w = np.array(frame["transform_matrix"], dtype=np.float32)

        if True:
            transform_matrix_c2w = transform_matrix_c2w[0:3, :]
            trans_c2w_ngp = nerf_matrix_to_ngp(transform_matrix_c2w, scale, offset)
            trans_c2w_ngp = np.vstack((trans_c2w_ngp, np.array([0.0,0.0,0.0,1.0])))
            # print("transform_matrix_c2w", transform_matrix_c2w)
            # print("trans_c2w_ngp", trans_c2w_ngp)
            camera_w2c = np.linalg.inv(trans_c2w_ngp)
        else:
            camera_w2c = np.linalg.inv(transform_matrix_c2w)


        # 单个相机内参数据
        int_dict = {}   # 单个相机的内参数据
        k_list = []
        for c in range(3):
            for r in range(3):
                k_list.append(np.float64(intrisic[r][c]))
        int_dict['intrinsic_matrix'] = k_list
        int_dict['height'] = height
        int_dict['width'] = width

        para_dict = {}
        rt_list = []
        for c in range(4):
            for r in range(4):
                rt_list.append(np.float64(camera_w2c[r][c]))
        para_dict['class_name'] = 'PinholeCameraParameters'
        para_dict['extrinsic'] = rt_list
        para_dict['intrinsic'] = int_dict
        para_dict['version_major'] = 1
        para_dict['version_minor'] = 0

        paras_list.append(para_dict)
    paras_dict['class_name'] = 'PinholeCameraTrajectory'
    paras_dict['parameters'] = paras_list

    with open(os.path.join(json_output, "para_open3d.json"), "w+") as f:
        json.dump(paras_dict, f)
        print("写入文件完成...")

def show_pose(neus_mesh_path, para_path, temp_path, width, height,render_option_path="/home/lhw/open3d_data/extract/DemoCustomVisualization/renderoption.json"):
    mesh = o3d.io.read_triangle_mesh(neus_mesh_path)

    custom_draw_geometry_with_camera_trajectory(
        mesh, render_option_path, para_path, temp_path,width, height)


def meshNormalReverse(neus_mesh_path, dst_neus_mesh_path):
    mesh = o3d.io.read_triangle_mesh(neus_mesh_path)
    for i in range(len(mesh.vertex_normals)):
        mesh.vertex_normals[i] = -mesh.vertex_normals[i]
    o3d.io.write_triangle_mesh(dst_neus_mesh_path, mesh)


def nerf2mvstex(json_path,raw_color_path, main_path, neus_mesh_path, neus_mesh_name, render_color_path=""):
    mesh_output_path = os.path.join(neus_mesh_path, neus_mesh_name) 
    texture_path = os.path.join(main_path , "texture/")
    os.makedirs(texture_path, exist_ok=True)

    rgb_cam_path = os.path.join(texture_path, "texture_data")
    new_ply_path = os.path.join(texture_path, neus_mesh_name)
    os.makedirs(rgb_cam_path, exist_ok=True)
    # shutil.copy(mesh_output_path, new_ply_path)
    # dst_ply_path = os.path.join(texture_path, "neus_mesh_n_reverse.ply")
    
    meshNormalReverse(mesh_output_path, new_ply_path)

    transform_json={}
    with open(json_path, 'r') as f:
        transform_json = json.load(f)
        # print(transform_json)
    scale = 1.0
    offset = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)

    # scale = 0.6
    # offset = np.array([[0.5], [0.5], [0.0]], dtype=np.float32)
    print("scale", scale)
    color_path = raw_color_path
    for frame in transform_json["frames"]:
        file_path = frame["file_path"]
        file_name = file_path[file_path.rfind('/')+1:]
        file_idx = file_name[0:file_name.rfind('.')]
        # print("file idx", file_idx)
        width = frame["w"]
        height = frame["h"]
        intrisic = np.array([[frame["fl_x"], 0.0, float(frame["cx"])],
                              [0.0, frame["fl_y"], float(frame["cy"])],
                              [0.0, 0.0, 1.0]
                              ],  dtype=np.float32)
        
        # angle_x = np.arctan(width / (frame["fl_x"] * 2)) * 2
        # angle_y = np.arctan(height / (frame["fl_y"] * 2)) * 2
        # fovx = angle_x * 180 / np.pi
        # fovy = angle_y * 180 / np.pi
        # print(fovx, fovy)

        distortion = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        transform_matrix_c2w = np.array(frame["transform_matrix"], dtype=np.float32)

    
        transform_matrix_c2w = transform_matrix_c2w[0:3, :]
        trans_c2w_ngp = nerf_matrix_to_ngp(transform_matrix_c2w, scale, offset)
        trans_c2w_ngp = np.vstack((trans_c2w_ngp, np.array([0.0,0.0,0.0,1.0])))
        # print("transform_matrix_c2w", transform_matrix_c2w)
        # print("trans_c2w_ngp", trans_c2w_ngp)
        camera_w2c = np.linalg.inv(trans_c2w_ngp)
        print(camera_w2c)
        # break


        # print()


        src_img_path = os.path.join(color_path, file_idx+".png")
        dst_img_path = os.path.join(rgb_cam_path, file_idx+".png")
        shutil.copy(src_img_path, dst_img_path)

        camera_w2c_string = str(camera_w2c[0, 3]) + " " + str(camera_w2c[1, 3]) + " " + str(camera_w2c[2, 3]) + " " \
                                                        + str(camera_w2c[0, 0]) + " " + str(camera_w2c[0, 1]) + " " + str(camera_w2c[0, 2]) + " "\
                                                        + str(camera_w2c[1, 0]) + " " + str(camera_w2c[1, 1]) + " " + str(camera_w2c[1, 2]) + " "\
                                                        + str(camera_w2c[2, 0]) + " " + str(camera_w2c[2, 1]) + " " + str(camera_w2c[2, 2]) + "\n"
        d0 = 0.0
        d1 = 0.0
        paspect = intrisic[1, 1] / intrisic[0, 0]
        dim_aspect = float(width) / float(height)
        img_aspect = dim_aspect * paspect
        if img_aspect < 1.0:
            f = intrisic[1, 1] / height
        else:
            f = intrisic[0, 0] / width
        ppx = intrisic[0, 2] / width
        ppy = intrisic[1, 2] / height
        intrisic_string = str(f) + " " + str(d0) + " " + str(d1) + " " + str(paspect) + " " + str(ppx) + " " + str(ppy) + "\n"
        cam_path = os.path.join(rgb_cam_path, file_idx+".cam")
        with open(cam_path, 'w') as file:
            file.write(camera_w2c_string)
            file.write(intrisic_string)
        print("cam_path:",cam_path)
    

def Nerf2MvstexWithRender(json_path, main_path, neus_mesh_path, neus_mesh_name, render_path):
    mesh_output_path = os.path.join(neus_mesh_path, neus_mesh_name) 
    texture_path = os.path.join(main_path , "texture/")
    os.makedirs(texture_path, exist_ok=True)

    rgb_cam_path = os.path.join(texture_path, "texture_data")
    new_ply_path = os.path.join(texture_path, neus_mesh_name)
    os.makedirs(rgb_cam_path, exist_ok=True)
    shutil.copy(mesh_output_path, new_ply_path)
    # dst_ply_path = os.path.join(texture_path, "neus_mesh_n_reverse.ply")
    
    # meshNormalReverse(new_ply_path, dst_ply_path)

    transform_json={}
    with open(json_path, 'r') as f:
        transform_json = json.load(f)
        # print(transform_json)
    scale = 1.0
    offset = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)

    # scale = 0.25
    # offset = np.array([[0.5], [0.5], [0.5]], dtype=np.float32)
    print("scale", scale)

    with open(os.path.join(render_path, "transforms_render.json"), 'r') as f:
        transform_json_render = json.load(f)
        print(os.path.join(render_path, "transforms_render.json"))
    frame_1_len = len(transform_json["frames"])
    frame_2_len = len(transform_json_render["frames"])
    frame_num = frame_1_len + frame_2_len
    for frame_idx in range(frame_num):
        if frame_idx < len(transform_json["frames"]):
            color_path = os.path.join(main_path, "color/")
            frame = transform_json["frames"][frame_idx]
            continue
        else:
            color_path = os.path.join(render_path, "color/")
            frame = transform_json_render["frames"][frame_idx-frame_1_len]
            if(frame_idx-frame_1_len > 0):
                continue
            print("file idx:",frame_idx-frame_1_len)
            # scale = 0.25
            # offset
        width = frame["w"]
        height = frame["h"]
        file_path = frame["file_path"]
        file_name = file_path[file_path.rfind('/')+1:]
        file_idx = file_name[0:file_name.rfind('.')]
        print("file idx", file_idx)

        intrisic = np.array([[frame["fl_x"], 0.0, float(frame["cx"])],
                              [0.0, frame["fl_y"], float(frame["cy"])],
                              [0.0, 0.0, 1.0]
                              ],  dtype=np.float32)
        print("intrisic", intrisic)
        angle_x = np.arctan(width / (frame["fl_x"] * 2)) * 2
        angle_y = np.arctan(height / (frame["fl_y"] * 2)) * 2
        fovx = angle_x * 180 / np.pi
        fovy = angle_y * 180 / np.pi
        print(fovx, fovy)

        distortion = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        transform_matrix_c2w = np.array(frame["transform_matrix"], dtype=np.float32)

    
        transform_matrix_c2w = transform_matrix_c2w[0:3, :]
        trans_c2w_ngp = nerf_matrix_to_ngp(transform_matrix_c2w, scale, offset)
        trans_c2w_ngp = np.vstack((trans_c2w_ngp, np.array([0.0,0.0,0.0,1.0])))
        # print("transform_matrix_c2w", transform_matrix_c2w)
        # print("trans_c2w_ngp", trans_c2w_ngp)
        camera_w2c = np.linalg.inv(trans_c2w_ngp)
        print(camera_w2c)
        # break


        print("camera_w2c:",camera_w2c)


        src_img_path = os.path.join(color_path, file_idx+".jpg")
        dst_img_path = os.path.join(rgb_cam_path, file_idx+".jpg")
        shutil.copy(src_img_path, dst_img_path)

        camera_w2c_string = str(camera_w2c[0, 3]) + " " + str(camera_w2c[1, 3]) + " " + str(camera_w2c[2, 3]) + " " \
                                                        + str(camera_w2c[0, 0]) + " " + str(camera_w2c[0, 1]) + " " + str(camera_w2c[0, 2]) + " "\
                                                        + str(camera_w2c[1, 0]) + " " + str(camera_w2c[1, 1]) + " " + str(camera_w2c[1, 2]) + " "\
                                                        + str(camera_w2c[2, 0]) + " " + str(camera_w2c[2, 1]) + " " + str(camera_w2c[2, 2]) + "\n"
        d0 = 0.0
        d1 = 0.0
        paspect = intrisic[1, 1] / intrisic[0, 0]
        dim_aspect = float(width) / float(height)
        img_aspect = dim_aspect * paspect
        if img_aspect < 1.0:
            f = intrisic[1, 1] / height
        else:
            f = intrisic[0, 0] / width
        ppx = intrisic[0, 2] / width
        ppy = intrisic[1, 2] / height
        intrisic_string = str(f) + " " + str(d0) + " " + str(d1) + " " + str(paspect) + " " + str(ppx) + " " + str(ppy) + "\n"
        cam_path = os.path.join(rgb_cam_path, file_idx+".cam")
        with open(cam_path, 'w') as file:
            file.write(camera_w2c_string)
            file.write(intrisic_string)
        print("cam_path:",cam_path)

if __name__ == "__main__":
    args = parse_args()
    root_path = args.root_path
    width = int(args.width)
    height = int(args.height)
    flag = int(args.flag)
    neus_mesh_name = args.neus_mesh_name
    transforms_path = os.path.join(root_path, "transforms.json")
    neus_mesh_path = args.neus_mesh_path

    if flag==2:
        json_output = os.path.join(root_path, "open3d")
        os.system("mkdir -p "+json_output)
        neus_mesh_path = os.path.join(neus_mesh_path, neus_mesh_name)
        Pose2Open3d(json_output, transforms_path)
        temp_path = os.path.join(json_output, "temp")
        os.system("mkdir -p "+temp_path)
        width = 2160
        height = 3840
        param_json = os.path.join(json_output, "para_open3d.json")
        show_pose(neus_mesh_path, param_json, temp_path, width, height)
    elif flag==0:
        color_path = os.path.join(args.render_dir, "rgba/")

        nerf2mvstex(transforms_path, color_path,root_path, neus_mesh_path, neus_mesh_name)
    elif flag==2:
        render_folder_path = args.render_dir

        Nerf2MvstexWithRender(transforms_path, root_path, neus_mesh_path, neus_mesh_name,render_folder_path)
    elif flag==3:
        transforms_path = os.path.join(args.render_dir, "transforms_render.json")
        raw_color_path = os.path.join(args.root_path, "color/")
        render_color_path = os.path.join(args.render_dir, "color/")


        nerf2mvstex(transforms_path, raw_color_path,root_path, neus_mesh_path, neus_mesh_name, render_color_path)

    