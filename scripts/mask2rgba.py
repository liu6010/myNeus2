import os
import argparse
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="mask to rgba")
    parser.add_argument("--color_path", default="", help="the input color images path")
    parser.add_argument("--mask_path", default="", help="the input mask images path")
    parser.add_argument("--output_path", default="", help="the output rgba images path")
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    color_path = args.color_path
    mask_path = args.mask_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    color_names = sorted(os.listdir(color_path))
    mask_names = sorted(os.listdir(mask_path))

    n = 0
    for color_names, mask_names in zip(color_names, mask_names):
        print("processing " + str(n + 1) + " image")
        color = cv2.imread(os.path.join(color_path, color_names))
        mask = cv2.imread(os.path.join(mask_path, mask_names), -1)
        fragement = np.zeros((np.shape(color)[0], np.shape(color)[1], 4))
        mask = mask / 255
        if len(mask.shape) == 3:
            fragement[:, :, 0] = color[:, :, 0] * (mask[:, :, 0] > 0)
            fragement[:, :, 1] = color[:, :, 1] * (mask[:, :, 1] > 0)
            fragement[:, :, 2] = color[:, :, 2] * (mask[:, :, 2] > 0)
            fragement[:, :, 3] = 0 * (mask[:, :, 0] == 0) + 255 * (mask[:, :, 0] != 0)
        else:
            fragement[:, :, 0] = color[:, :, 0] * (mask[:, :] > 0)
            fragement[:, :, 1] = color[:, :, 1] * (mask[:, :] > 0)
            fragement[:, :, 2] = color[:, :, 2] * (mask[:, :] > 0)
            fragement[:, :, 3] = 0 * (mask[:, :] == 0) + 255 * (mask[:, :] != 0)
        save_fragement = np.asarray(fragement, dtype=np.uint8)
        cv2.imwrite(os.path.join(output_path, color_names[:-3] + "png"), save_fragement)
        n = n + 1
