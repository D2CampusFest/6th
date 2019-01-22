"""
MobileNet V2으로 이미지 특징 벡터를 추출하는 모듈입니다.
"""
import os
import time
import numpy as np
import tensorflow as tf
from mobilenet_v2 import MobileNet2, get_encoded_image
from config import *


def extract_features():
    """
    IMG_DIR에 있는 모든 이미지에 대해 MobineNet V2 특징 벡터를 추출합니다.
    추출된 특징 벡터는 DATA_DIR/FEATURES.npy 에 저장됩니다.
    BATCH_SIZE로 배치 사이즈를 조절할 수 있습니다.
    :return: 없음
    """
    # get list all images
    img_paths = os.listdir(IMG_DIR)
    img_paths.sort()
    img_paths = [os.path.join(IMG_DIR, filename) for filename in img_paths if filename.endswith(IMG_EXT)]
    with open(os.path.join(DATA_DIR, IMG_PATHS), 'w') as f:
        f.writelines([line + "\n" for line in img_paths])

    # prepare tf.dataset for batch inference
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.map(lambda img_path:
                          (img_path, tf.py_func(get_encoded_image, [img_path], [tf.string])))
    batched_dataset = dataset.batch(BATCH_SIZE)
    iterator = batched_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    # build dnn model
    model = MobileNet2()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # batch inference images
    num_processed_images = 0
    features = np.ndarray(shape=[0, model.output_size])
    while True:
        try:
            start_time = time.process_time()
            # get batch of encoded images
            batch = sess.run(next_batch)
            batched_img_paths = batch[0]
            batched_encoded_images = batch[1]
            cur_batch_size = len(batched_img_paths)
            # get batch of features
            batched_features = sess.run(model.features, feed_dict={
                model.filename: batched_img_paths,
                model.encoded_images: np.reshape(batched_encoded_images, [cur_batch_size])})
            features = np.concatenate((features, batched_features))
            # show progress
            num_processed_images += len(batched_encoded_images)
            elapsed_time = time.process_time() - start_time
            print("Processed images: %d,\tElapsed time per img: %.2f" % (num_processed_images, elapsed_time / BATCH_SIZE))
        except tf.errors.OutOfRangeError:
            break

    # save npy and tsv files
    if os.path.exists(DATA_DIR) is False:
        os.makedirs(DATA_DIR)
    np.save(os.path.join(DATA_DIR, FEATURES + ".npy"), features)
    np.savetxt(os.path.join(DATA_DIR, FEATURES + ".tsv"), features, delimiter="\t")


if __name__ == '__main__':
    extract_features()
