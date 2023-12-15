import cv2
import numpy as np
import argparse
import os


def args():
    argparser = argparse.ArgumentParser(description="the depth erode")

    argparser.add_argument("--depth", help="the depth dir")
    argparser.add_argument("--iter", type=int, help="the iterations")
    argparser.add_argument("--save", help="the result depth save dir")

    args = argparser.parse_args()

    return args


if __name__ == '__main__':
    args = args()
    depth_dir = args.depth
    iter = args.iter
    save_dir = args.save

    names = os.listdir(depth_dir)

    for name in names:
        depth_path = os.path.join(depth_dir, name)
        save_path = os.path.join(save_dir, name)
        raw_path = os.path.join(save_dir, name[:-4]+"_raw.png")

        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        binary_image = (depth[:] > np.ones((depth.shape[0], depth.shape[1]))*1000)*255
        binary_image = np.asarray(binary_image, np.uint8)

        show_image = np.zeros((depth.shape[0], depth.shape[1], 3))
        show_image[:, :, 0] = binary_image
        show_image[:, :, 1] = binary_image
        show_image[:, :, 2] = binary_image

        kernel = np.ones((3,3), np.uint8)
        result_bin = cv2.erode(show_image, kernel, iterations=iter)

        raw_depth = binary_image/255 * depth
        raw_depth = np.asarray(raw_depth, np.uint16)
        result_depth = result_bin[:,:,0]/255 * depth
        result_depth = np.asarray(result_depth, np.uint16)

        cv2.imwrite(raw_path, raw_depth)
        cv2.imwrite(save_path, result_depth)
