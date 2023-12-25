#!/bin/bash
source /home/fhy/anaconda3/etc/profile.d/conda.sh
data_path="./data/person_ue4"
gaussian_code_path=/home/fhy/workspace/lhw/gaussian-splatting-zy

param_path=$data_path/xml/
texture_path=$data_path/texture/
rgba_path=$data_path/rgba

render_file=render_1224_test
gaussian_out_path=${data_path}/gaussian/gaussian_$render_file


out_name=person_ue4
out_name_new=${out_name}_25w_new


image_num=39
width=2160
height=3840
RootPath=/home/fhy/workspace/lhw/NeuS2/
conda activate neus2

n_steps=200000
date=1215
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
            --scale 0.22 \
            --use_depth True \
            --w $width \
            --h $height \
            --out $RootPath/$data_path/transforms.json
    ;;
    "1-1-1"):
        cd $data_path
        python ../../scripts/xml2nerf_multi.py \
            --images rgba/%04d.png \
            --xml_in xml_ue4/%04d.xml \
            --depths depth/%04d.png \
            --n $image_num \
            --aabb_scale 2 \
            --scale 0.22 \
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
    
    "2")
        ./build/testbed --scene ${data_path}/transforms.json
        # python scripts/run.py --scene ${data_path}/transforms.json -m nerf --snapshot ${data_path}/save.msgpack --n_steps 15000
        # python scripts/run.py --scene ./data/1/transforms.json -m nerf --snapshot ./data/1/save.msgpack --n_steps 15000 --name person1
        # python scripts/run_color_by_path.py \
        #     --screenshot_transforms /home/lhw/Gradute/RenderAndRecon/NeuS2/data/person2/expirement/1102/transforms_neus2_depth00_cam.json
    ;;
    "2-0")
        ./build/testbed --scene $data_path/transforms.json
    ;;
    "2-1")
        n_steps=75000
        out_name=${out_name}_0.8
        out_name_new=${out_name}_75k
        echo $out_name_new
        python ./scripts/run_ModelMesh.py \
            --scene $data_path/transforms.json \
            --name ${out_name_new} \
            --n_steps $n_steps \
            --marching_cubes_res 800
            # --optimize_exposure
    ;;
    "2-2")
        out_name_new=${out_name}_20w
        echo $out_name_new
        python ./scripts/run_ModelMesh.py \
            --scene $data_path/transforms.json \
            --name ${out_name_new} \
            --n_steps $n_steps \
            --marching_cubes_res 800 \
            --optimize_exposure
    ;;
    "sh")
        for i in $(seq 0.2 0.2 1)
        do 
            out_name_new=${out_name}_$i
            echo $out_name_new
            python ./scripts/run_ModelMesh.py \
                --scene $data_path/transforms.json \
                --name ${out_name_new} \
                --n_steps $n_steps \
                --optimize_exposure \
                --depth_supervision_lambda $i
        done

        

    ;;
    "4-1")
        mkdir -p ./$data_path/render/$render_file/color_neus2/
        python ./scripts/transformsInterpolate.py \
            --transforms_path ${data_path}/transforms.json --data_name $out_name \
            --out_transforms_path ${data_path}/render/$render_file/transforms_render.json
        python ./scripts/render_video_by_path.py \
            --scene ${data_path}/transforms.json --mode nerf \
            --load_snapshot $data_path/output/$out_name_new/checkpoints/75000.msgpack \
            --width $width --height $height --render_mode shade \
            --screenshot_transforms ${data_path}/render/$render_file/transforms_render.json \
            --screenshot_dir ./$data_path/render/$render_file/color_neus2/
            # --screenshot_dir ./$data_path/render/color_temp/

    ;;
    "4-2")
        
        # mkdir -p $gaussian_out_path/color
        mkdir -p $gaussian_out_path/bk

        # cp ${data_path}/render/transforms_render.json $gaussian_out_path/bk
        # cp ${data_path}/transforms.json ${gaussian_out_path}/bk
        cp -r ${data_path}/rgba $gaussian_out_path

        python ./scripts/transforms2gaussian.py \
            --transforms ${data_path}/transforms.json \
            --gaussian $gaussian_out_path/transforms.json

        python ./scripts/transforms2gaussian.py \
            --transforms ${data_path}/render/$render_file/transforms_render.json \
            --gaussian $gaussian_out_path/gaussian_$render_file.json

        python ./scripts/exportMeshToPcl.py \
            --mesh_path $data_path/output/$out_name_new/mesh/75000.ply \
            --out_pcl_path $gaussian_out_path/points3d.ply
    ;;
    "4-3")
        conda activate gaussian_splatting
        code_path=/home/fhy/workspace/lhw/gaussian-splatting-zy

        epoch=20000

        python $code_path/train.py -s $gaussian_out_path -r 1 \
                 --iterations $epoch \
                 -m $gaussian_out_path/output --data_device cpu  
        
        python $code_path/render.py \
            -s $gaussian_out_path \
            -m $gaussian_out_path/output

    ;;
    "4-4")
        conda activate gaussian_splatting
        python $gaussian_code_path/render_by_path.py --quiet \
            --track_path $gaussian_out_path/gaussian_$render_file.json \
            --save_path  $gaussian_out_path/render \
            -s $gaussian_out_path \
            -m $gaussian_out_path/output \
            --width $width \
            --height $height
    ;;


esac


