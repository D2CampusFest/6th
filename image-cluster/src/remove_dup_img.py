"""
중복 이미지를 제거하는 모듈입니다.
"""
import os
import re
import cv2
import img_utils
from config import *


def remove_dup_img():
    re_model = re.compile("^(\d+)_")

    img_paths = os.listdir(IMG_DIR)
    img_paths.sort()

    labels = [re_model.match(img_path).group(1) for img_path in img_paths if img_path.endswith(IMG_EXT)]
    unique_labels = list(set(labels))
    dup_img_indices = []

    for label in unique_labels:
        img_indices = [i for i, v in enumerate(labels) if v == label]
        if len(img_indices) < 2:
            continue
        img_id_list = [(img_id, cv2.imread(os.path.join(IMG_DIR, img_paths[img_id]))) for img_id in img_indices]
        unique_img_list = [img_id_list[0][1]]
        unique_img_id_list = [img_id_list[0][0]]
        for i in range(1, len(img_id_list)):
            found = img_utils.find_first_same_image_in_list(unique_img_list, img_id_list[i][1])
            if found >= 0:
                # print(files[img_id_list[i][0]], " == ", files[unique_img_id_list[found]])
                dup_img_indices.append(img_id_list[i][0])
            else:
                unique_img_id_list.append(img_id_list[i][0])
                unique_img_list.append(img_id_list[i][1])

    print("Number of dup images: ", len(dup_img_indices))

    [os.remove(os.path.join(IMG_DIR, img_paths[i]))
     for i in dup_img_indices
     if os.path.exists(os.path.join(IMG_DIR, img_paths[i]))]


if __name__ == '__main__':
    remove_dup_img()
