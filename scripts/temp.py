import os
import shutil

root_path = "/home/fhy/new_disk1/lhw/NeuS2/output/person_test/images"
out_path = "/home/fhy/new_disk1/lhw/NeuS2/output/person_test/render"

os.makedirs(out_path, exist_ok=True)



num = 39
for i in range(num):
    dir_name = str(i).zfill(4)
    src_path = os.path.join(root_path,dir_name, "frame_000000_pred.png")
    dst_path = os.path.join(out_path, dir_name+".png")

    shutil.copy(src_path, dst_path)

