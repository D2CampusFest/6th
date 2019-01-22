"""
이미지 폴더의 파일을 분석하여 정답 레이블(labels_true)을 생성하는 모듈입니다.
"""
import os
import re
import numpy as np
from config import *


def make_labels_true():
    """
    이미지 폴더의 파일을 분석하여 정답 레이블(labels_true)을 생성하는 함수입니다.
    정답 레이블에 대응하는 이미지 파일 경로는 IMG_PATHS 에 저장됩니다.
    :return: None
    """
    # regex rule to extract true labels
    re_model = re.compile("^(\d+)_")

    # get list of files
    if os.path.exists(IMG_DIR) is False:
        print("Folder %s not found, copy image files to %s" % (IMG_DIR, IMG_DIR))
        return
    img_paths = os.listdir(IMG_DIR)
    img_paths.sort()
    img_paths = [filename for filename in img_paths if filename.endswith(IMG_EXT)]

    labels_true = [re_model.match(img_path).group(1) for img_path in img_paths]

    print("Total number of images: %d, total number of models: %d" % (len(labels_true), len(set(labels_true))))

    if os.path.exists(DATA_DIR) is False:
        os.makedirs(DATA_DIR)

    with open(os.path.join(DATA_DIR, IMG_PATHS), 'w') as f:
        f.writelines([line + "\n" for line in img_paths])
    with open(os.path.join(DATA_DIR, LABELS_TRUE), 'w') as f:
        f.writelines([line + "\n" for line in labels_true])

    # additionally save npy and tsv
    np.save(os.path.join(DATA_DIR, LABELS_TRUE + ".npy"), labels_true)
    np.savetxt(os.path.join(DATA_DIR, LABELS_TRUE + ".tsv"), labels_true, "%s", delimiter="\t")


if __name__ == '__main__':
    make_labels_true()
