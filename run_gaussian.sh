#!/bin/bash
source /home/fhy/anaconda3/etc/profile.d/conda.sh
data_path="./data/person_gn"
param_path=$data_path/xml/
texture_path=$data_path/texture/
rgba_path=$data_path/rgba
neus2_mesh_name=neus2_mesh_depth00_1106.ply
gaussian_code_path=/home/fhy/workspace/lhw/gaussian-splatting-zy

render_file=render_1224_test

out_name=person_gn
out_name_new=${out_name}_false_75k_new

render_file=render_1227_test
gaussian_out_path=${data_path}/gaussian/gaussian_$render_file

image_num=39
width=2160
height=3840
RootPath=/home/fhy/workspace/lhw/NeuS2/
conda activate neus2

n_steps=25000
date=1217
case $1 in

    "4-3")
        conda activate gaussian_splatting
        code_path=/home/fhy/workspace/lhw/gaussian-splatting-zy
        epoch=20000

        data_path="./data/person_gn"
        render_file=render_1228_test
        gaussian_out_path=${data_path}/gaussian/gaussian_$render_file
        python $code_path/train.py -s $gaussian_out_path -r 1 \
                 --iterations $epoch --save_iterations $epoch \
                 -m $gaussian_out_path/output --data_device cpu  
        python $gaussian_code_path/render_by_path.py --quiet \
            --track_path $gaussian_out_path/transforms.json \
            --save_path  $gaussian_out_path/render_test \
            -s $gaussian_out_path \
            -m $gaussian_out_path/output \
            --width $width \
            --height $height
        
        data_path="./data/person2"
        render_file=render_1228_test
        gaussian_out_path=${data_path}/gaussian/gaussian_$render_file
        python $code_path/train.py -s $gaussian_out_path -r 1 \
                 --iterations $epoch --save_iterations $epoch \
                 -m $gaussian_out_path/output --data_device cpu  
        python $gaussian_code_path/render_by_path.py --quiet \
            --track_path $gaussian_out_path/transforms.json \
            --save_path  $gaussian_out_path/render_test \
            -s $gaussian_out_path \
            -m $gaussian_out_path/output \
            --width $width \
            --height $height
        
        data_path="./data/person_ue4"
        render_file=render_1228_test
        gaussian_out_path=${data_path}/gaussian/gaussian_$render_file
        python $code_path/train.py -s $gaussian_out_path -r 1 \
                 --iterations $epoch --save_iterations $epoch \
                 -m $gaussian_out_path/output --data_device cpu  
        python $gaussian_code_path/render_by_path.py --quiet \
            --track_path $gaussian_out_path/transforms.json \
            --save_path  $gaussian_out_path/render_test \
            -s $gaussian_out_path \
            -m $gaussian_out_path/output \
            --width $width \
            --height $height

    ;;


esac


