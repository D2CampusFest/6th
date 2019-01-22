"""
Image Match 의 ImageSignature 모듈로 이미지 시그니쳐 벡터를 추출하는 모듈입니다.
"""
import os
import numpy as np
from image_match.goldberg import ImageSignature
from tqdm import tqdm
from config import *


def extract_signature():
    img_paths = os.listdir(IMG_DIR)
    img_paths.sort()
    img_paths = [os.path.join(IMG_DIR, filename) for filename in img_paths if filename.endswith(IMG_EXT)]
    with open(os.path.join(DATA_DIR, IMG_PATHS), 'w') as f:
        f.writelines([line + "\n" for line in img_paths])

    # init a signature generator
    gis = ImageSignature()

    # process images
    num_processed_images = 0
    signatures = np.ndarray(shape=[0, gis.sig_length])

    for img_path in tqdm(img_paths):
        sig = gis.generate_signature(img_path)
        signatures = np.concatenate((signatures, np.reshape(sig, (1, gis.sig_length))))

    # save signatures to npy file
    if os.path.exists(DATA_DIR) is False:
        os.makedirs(DATA_DIR)
    np.save(os.path.join(DATA_DIR, SIGNATURES), signatures)


if __name__ == '__main__':
    extract_signature()
