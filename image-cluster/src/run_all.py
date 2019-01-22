"""
전체 프로세스를 보여주는 모듈입니다.
"""
from make_labels_true import *
from extract_features import *
from make_labels_pred import *
from evaluation import *
from visualize import *


if __name__ == '__main__':
    if os.path.exists(IMG_DIR):
        # make true labels by analysing image filename
        make_labels_true()

        # extract image features using MobileNet V2
        extract_features()

        # make cluster using K-Means algorithm
        make_labels_pred()

        # evaluate clustering result by adjusted Rand index
        evaluation()

        # visualize clustering using t-SNE
        visualize()
    else:
        print("Image dir not found.")
        pass
