#!/bin/bash
source /home/fhy/anaconda3/etc/profile.d/conda.sh
data_path="./data/person_gn"
param_path=$data_path/xml/
texture_path=$data_path/texture/
rgba_path=$data_path/rgba
neus2_mesh_name=neus2_mesh_depth00_1106.ply
# render_dir=gaussian_render1107
render_dir=gaussian_render_1108

image_num=39
width=2160
height=3840
RootPath=/home/fhy/workspace/lhw/NeuS2/
conda activate neus2
out_name=person_gn
n_steps=10000
date=1217
case $1 in
    "0"):
        cd $RootPath
        python ./scripts/mask2rgba.py \
            --color_path /home/lhw/Gradute/3Dphoto/scan_obj_e2/data/neus2Data/color/ \
            --mask_path /home/lhw/Gradute/3Dphoto/scan_obj_e2/data/neus2Data/mask \
            --output_path /home/lhw/Gradute/3Dphoto/scan_obj_e2/data/neus2Data/rgba/
    ;;
    "1-0"):
        python ./scripts/neus2_render_path.py \
            --transforms $data_path/transforms.json \
            --out_transforms $data_path/transforms_render.json
    ;;
    "1-1"):
        cd $data_path
        python ../../scripts/xml2nerf_multi_stereo.py \
            --images rgba/%04d.png \
            --xml_in xml/%04d.xml \
            --depths depth/%04d.png \
            --n $image_num \
            --aabb_scale 2 \
            --scale 0.25 \
            --use_depth True \
            --w $width \
            --h $height \
            --out $RootPath/$data_path/transforms.json
    ;;
    "1-1-2"):
        data_path=/home/lhw/Gradute/RenderAndRecon/NeuS2/data/person_gn2/
        cd $data_path
        python ../../scripts/xml2gaussian_multi_keepcolmap.py \
            --images rgba/%04d.png \
            --xml_in xml/%04d.xml \
            --n $image_num \
            --aabb_scale 2 \
            --scale 0.25 \
            --use_depth False \
            --w $width \
            --h $height \
            --out $data_path/gaussian/transforms.json
    ;;
    "1-2"):
        conda activate 3dRecons
        cd $data_path
        rm -rf $texture_path/texture_data/*
        rm -rf $texture_path/temp_mask/*
        python ../../scripts/neus2_2_mvstex.py \
            --root_path $data_path \
            --width $width \
            --height $height \
            --neus_mesh_path ./expirement/1106/ \
            --neus_mesh_name $neus2_mesh_name \
            --flag 0
        conda deactivate
    ;;
    "1-3"):
        conda activate 3dRecons
        cd $data_path
        rm -rf $texture_path/texture_data/*
        rm -rf $texture_path/temp_mask/*
        python ../../scripts/neus2_2_mvstex.py \
            --root_path $data_path \
            --width $width \
            --height $height \
            --neus_mesh_path ./expirement/1106/ \
            --neus_mesh_name $neus2_mesh_name \
            --render_dir ./render/$render_dir \
            --flag 3
        conda deactivate
    ;;

    "1-4"):
        rm -rf $texture_path/output
        mkdir -p $texture_path/output
        rm -rf $texture_path/temp_mask/*
        # /home/lhw/Gradute/digitalManCode/Recons/mvs-texturing/build/apps/texrecon/texrecon \
        #     $texture_path/texture_data/ \
        #     $texture_path/$neus2_mesh_name \
        #     $texture_path/output/out/ \
        #     --keep_unseen_faces
        /home/lhw/Gradute/digitalManCode/Recons/mvs-texturing/build/apps/texrecon/texrecon \
            $texture_path/texture_data/ \
            $texture_path/neus2_mesh_depth00_1106_smooth.ply \
            $texture_path/output/out/
            # --keep_unseen_faces

            
    ;;
    "1-5"):
        rm -rf /home/lhw/Gradute/3Dphoto/scan_obj_e2/data/texture/output
        mkdir -p /home/lhw/Gradute/3Dphoto/scan_obj_e2/data/texture/output
        /home/lhw/Gradute/digitalManCode/Recons/mvs-texturing/build/apps/texrecon/texrecon \
            /home/lhw/Gradute/3Dphoto/scan_obj_e2/data/texture/texture_data/ \
            /home/lhw/Gradute/3Dphoto/scan_obj_e2/data/texture/tsdf_mesh.ply \
            /home/lhw/Gradute/3Dphoto/scan_obj_e2/data/texture/output/out/ \
            --keep_unseen_faces
    ;;
    
    "2"):
        # ./build/testbed --scene ${data_path}/transforms.json
        # python scripts/run.py --scene ${data_path}/transforms.json -m nerf --snapshot ${data_path}/save.msgpack --n_steps 15000
        # python scripts/run.py --scene ./data/1/transforms.json -m nerf --snapshot ./data/1/save.msgpack --n_steps 15000 --name person1
        python scripts/run_color_by_path.py \
            --screenshot_transforms /home/lhw/Gradute/RenderAndRecon/NeuS2/data/person2/expirement/1102/transforms_neus2_depth00_cam.json
    ;;
    "2-0")
        # ./build/testbed --scene $data_path
        ./build/testbed --scene $data_path/transforms.json
    ;;
    "2-1")
        out_name_new=${out_name}_false_1w
        echo $out_name_new
        python ./scripts/run_ModelMesh.py \
            --scene $data_path/transforms.json \
            --name ${out_name_new} \
            --n_steps $n_steps \
            --marching_cubes_res 800
        n_steps=5000
        out_name_new=${out_name}_false_5k
        echo $out_name_new
        python ./scripts/run_ModelMesh.py \
            --scene $data_path/transforms.json \
            --name ${out_name_new} \
            --n_steps $n_steps \
            --marching_cubes_res 800
    ;;
    "2-2")
        out_name_new=${out_name}_0.0_75k
        echo $out_name_new
        python ./scripts/run_ModelMesh.py \
            --scene $data_path/transforms.json \
            --name ${out_name_new} \
            --n_steps $n_steps \
            --marching_cubes_res 800 \
            --depth_supervision_lambda 0.0
    ;;
    "sh")
        for i in $(seq 0.0 0.2 1)
        do
            out_name_new=${out_name}_$i
            echo $out_name_new
            python ./scripts/run_ModelMesh.py \
                --scene $data_path/transforms.json \
                --name ${out_name_new} \
                --n_steps $n_steps \
                --marching_cubes_res 800 \
                --depth_supervision_lambda $i
        done
    ;;
    "3-1"):
        conda activate 3dRecons
        cd $data_path
        rm -rf $texture_path/texture_data/*
        rm -rf $texture_path/temp_mask/*
        python ../../scripts/neus2_2_mvstex.py \
            --root_path $data_path \
            --width $width \
            --height $height \
            --neus_mesh_path ./expirement/1106/ \
            --neus_mesh_name $neus2_mesh_name \
            --flag 0
        conda deactivate
    ;;

esac


