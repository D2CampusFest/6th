"""
이미지 폴더의 파일을 분석하여 예측 레이블(labels_pred)을 생성하는 모듈입니다.
"""
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from config import *


def make_labels_pred():
    """
    K-Means 알고리즘으로 특징 벡터를 클러스터링 하는 함수입니다.
    예측 레이블은 DATA_DIR/LABELS_PRED.npy 에 저장됩니다.
    :return: None
    """
    # load datasets
    features = np.load(os.path.join(DATA_DIR, FEATURES + ".npy"))

    # estimate number of clusters
    num_clusters = int(len(features)/NUM_IMGS_PER_MODEL)
    print("Estimated num_clusters: %d" % num_clusters)

    # make prediction
    labels_pred = KMeans(n_clusters=num_clusters, verbose=0).fit_predict(features)

    # save predicted labels
    np.save(os.path.join(DATA_DIR, LABELS_PRED + ".npy"), labels_pred)
    np.savetxt(os.path.join(DATA_DIR, LABELS_PRED + ".tsv"), labels_pred, "%d", delimiter="\t")
    np.savetxt(os.path.join(DATA_DIR, LABELS_PRED + ".txt"), labels_pred, "%d", delimiter="\t")


if __name__ == '__main__':
    make_labels_pred()