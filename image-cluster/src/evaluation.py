"""
정답 레이블과 예측 레이블 파일을 읽어서 Adjusted Rand index를 계산하여 반환하는 모듈입니다.
"""
import os
import numpy as np
from sklearn import metrics
from config import *


def evaluation(labels_true_path, labels_pred_path):
    """
    정답 레이블과 예측 레이블 파일을 읽어서 Adjusted Rand index를 계산하여 반환하는 함수입니다.

    :param labels_true_path: 정답 이미지 레이블 파일
    labels_true의 예:
    01
    01
    01
    02
    02
    03
    ...

    :param labels_pred_path: 예측 이미지 레이블 파일
    labels_pred의 예:
    01
    01
    02
    02
    03
    04
    ...

    :return: Adjust Rand index (0~1) 계산하여 출력하고 점수를 반환함. 1에 가까울수록 정확
    예:
    Rand Index: 0.18918918918918917

    참고: https://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-index
    """
    # loading labels
    labels_true = np.loadtxt(labels_true_path, dtype=str)
    labels_pred = np.loadtxt(labels_pred_path, dtype=str)

    # compare labels
    return metrics.adjusted_rand_score(labels_true, labels_pred)


if __name__ == '__main__':
    """
    평가 예시:
    labels_pred1, labels_pred2 두개 가상의 레이블 파일을 각각 평가해 봅니다.
    labels_ture     01, 01, 01, 02, 02, 03 
    labels_pred1    01, 01, 02, 02, 03, 04
    labels_pred2    01, 01, 01, 02, 03, 04
    
    실행한 결과는 다음과 같습니다.  
    Score for labels_pred1.txt: 0.18918918918918917
    Score for labels_pred2.txt: 0.8148148148148149
    
    labels_pred2 가 좀 더 정확하기 때문에 점수가 높게 평가 됩니다.
    """
    score = evaluation(os.path.join(DATA_DIR, LABELS_TRUE + ".txt"), os.path.join(DATA_DIR, LABELS_PRED + ".txt"))
    print("Rand Index: %s" % score)
