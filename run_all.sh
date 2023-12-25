#!/bin/bash

source /home/fhy/anaconda3/etc/profile.d/conda.sh
conda activate neus2


case $1 in
    "1")
        # data_name=person_gn
        # data_path="./data/${data_name}"
        # n_steps=75000
        # out_name_new=${data_name}_false_75k_new
        # echo $out_name_new
        # python ./scripts/run_ModelMesh.py \
        #     --scene $data_path/transforms.json \
        #     --name ${out_name_new} \
        #     --n_steps $n_steps \
        #     --marching_cubes_res 800
        
        # data_name=person_ue4
        # data_path="./data/${data_name}"
        # n_steps=250000
        # out_name_new=${data_name}_25w_new
        # echo $out_name_new
        # python ./scripts/run_ModelMesh.py \
        #     --scene $data_path/transforms.json \
        #     --name ${out_name_new} \
        #     --n_steps $n_steps \
        #     --marching_cubes_res 800 \
        #     --optimize_exposure

        data_name=person2
        data_path="./data/${data_name}"
        n_steps=75000
        out_name_new=${data_name}_75k_new
        echo $out_name_new
        python ./scripts/run_ModelMesh.py \
            --scene $data_path/transforms.json \
            --name ${out_name_new} \
            --n_steps $n_steps \
            --optimize_exposure \
            --marching_cubes_res 800 
        

    #     data_name=person1
    #     data_path="./data/${data_name}"
    #     n_steps=75000
    #     out_name_new=${data_name}_75k_new
    #     echo $out_name_new
    #     python ./scripts/run_ModelMesh.py \
    #         --scene $data_path/transforms.json \
    #         --name ${out_name_new} \
    #         --n_steps $n_steps \
    #         --optimize_exposure \
    #         --marching_cubes_res 800 
        
    # ;;
    

esac
