#!/bin/bash

source /home/fhy/Software/anaconda3/etc/profile.d/conda.sh
data_root_path=data
data_name=person_gn
data_path=$data_root_path/$data_name
color_path=${data_path}/color
mask_path=${data_path}/mask
rgba_path=${data_path}/rgba
output_path=${data_path}/output
render_file=render_1108
width=2160
height=3840
train_epoch=15000
image_num=39

conda activate neus2


case $1 in
    "1")
        cd ${data_path}
        python ../../scripts/xml2nerf_multi_stereo.py \
        --images rgba/%04d.png \
        --n $image_num \
        --use_depth True \
        --xml_in xml/%04d.xml \
        --depths depth/%04d.png \
        --aabb_scale 2 \
        --scale 0.25 \
        --w $width --h $height \
        --out transforms.json
    ;;
    "1-2")
        out_path=${data_path}/gaussian/gaussian_$render_file
        mkdir -p $out_path/color
        cp ${data_path}/render/$render_file/transforms_render.json $out_path/transforms_render.json
        python ./scripts/transforms2gaussian.py \
        --transforms ${data_path}/render/$render_file/transforms_render.json \
        --gaussian $out_path/gaussian_$render_file.json
    ;;

    "2")
        ./build/testbed --scene ./$data_path/transforms.json
        # python scripts/run.py --scene ./$data_path/transforms.json --name person_test --n_steps 5000 --mode nerf ----save_snapshot ./$data_path/expirement/1102/neus2_depth1$depth_super.msgpack
    ;;
    "2-1")
        ./build/testbed --scene ./$data_path/transforms_render.json --snapshot ./$data_path/expirement/1104/neus2_depth00.msgpack
        # ./build/testbed --scene ./$data_path/transforms.json --snapshot ./$data_path/expirement/1102/neus2_depth00.msgpack
        # python scripts/run.py --scene ./$data_path/transforms.json --name person2 \
        #     --load_snapshot ./$data_path/expirement/1102/neus2_depth00.msgpack \
        #     --save_mesh_path ./$data_path/transforms_base_dep00.ply \
        #     --marching_cubes_res 800
    ;;
    "2-2")
        depth_super=0
        python scripts/run_save_mesh.py --scene ./$data_path/transforms.json \
            --n_steps 15000 \
            --mode nerf \
            --save_mesh_path ./$data_path/expirement/1106/neus2_mesh_depth0$depth_super.ply \
            --save_snapshot ./$data_path/expirement/1106/neus2_depth0$depth_super.msgpack \
            --marching_cubes_res 800 \
            --depth_supervision_lambda 0.$depth_super
    ;;
    "2-3")
        python scripts/run_color_by_path.py \
            --scene ./$data_path/transforms.json \
            --load_snapshot ./$data_path/expirement/1102/neus2_depth00.msgpack \
            --npg_camera_path ./$data_path/render/transforms_neus2_depth00_cam.json \
            --screenshot_transforms_out ./$data_path/render/transforms_render.json \
            --screenshot_dir ./$data_path/render/color/ \
            --width 1920 --height 1920 \
            --render_mode shade

    ;;
    "2-4")
        # python scripts/neus2_render_path_gn.py
        python ./scripts/transformsInterpolate.py \
            --transforms_path ${data_path}/transforms.json \
            --out_transforms_path ${data_path}/render/transforms_render.json
        python ./scripts/render_video_by_path.py \
            --scene ${data_path}/transforms.json --mode nerf \
            --load_snapshot ./$data_path/expirement/1106/neus2_depth00.msgpack \
            --width 2160 --height 3840 --render_mode shade \
            --screenshot_transforms ${data_path}/render/transforms_render.json \
            --screenshot_dir ./$data_path/render/render_1108/color/
            # --screenshot_dir ./$data_path/render/color_temp/

    ;;
    "2-5")
        # python scripts/neus2_render_path_gn.py
        mkdir -p ./$data_path/render/$render_file/color/
        python ./scripts/transformsInterpolate.py \
            --transforms_path ${data_path}/transforms.json \
            --out_transforms_path ${data_path}/render/$render_file/transforms_render.json
        python ./scripts/render_video_by_path.py \
            --scene ${data_path}/transforms.json --mode nerf \
            --load_snapshot ./$data_path/expirement/1106/neus2_depth00.msgpack \
            --width 2160 --height 3840 --render_mode shade \
            --screenshot_transforms ${data_path}/render/$render_file/transforms_render.json \
            --screenshot_dir ./$data_path/render/$render_file/color/
            # --screenshot_dir ./$data_path/render/color_temp/

    ;;
    "2-8")
        python ./scripts/run_trueview.py \
            --scene ${data_path}/transforms.json --mode nerf \
            --load_snapshot ./$data_path/expirement/1104/neus2_depth00.msgpack \
            --test --name person_test
    ;;
esac

# 生成分割图RGBA
# python scripts/mask2rgba.py --color_path ${color_path} --mask_path ${mask_path} --output_path ${rgba_path}

# 生成transforms.json
# cd ${data_path}
# python ../../scripts/colmap2nerf_multi.py --images rgba \
# --use_depth False \
# --integer_depth_scale 0.001 \
# --text param \
# --aabb_scale 1 \
# --out transforms.json








