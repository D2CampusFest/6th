"""
폴더와 임시 파일을 지정하는 Constants 들입니다.
"""

# dataset
DATASET = "test"                        # img 폴더 아래 폴더를 새로 만들고 해당 폴더이름으로 바꿔야 함

# image format
IMG_EXT = "jpg"                         # 이미지 파일 형식

# output dirs
IMG_DIR = "../img/" + DATASET           # 이미지 파일 경로
DATA_DIR = "../data/" + DATASET         # 중간 결과 경로
CLUSTER_DIR = "../result/" + DATASET    # 클러스터링 결과 경로

# files generated
IMG_PATHS = "img_paths.txt"             # 이미지 파일 경로
FEATURES = "features"                   # 이미지 특징 벡터
SIGNATURES = "signature"                # 이미지 시그니쳐
LABELS_TRUE = "labels_true"             # 정답 레이블
LABELS_PRED = "labels_pred"             # 예측 레이블

# for mobilenet feature extraction
BATCH_SIZE = 128                        # MobileNet V2 batch size

# for clustering
NUM_IMGS_PER_MODEL = 70                 # 클러스터당 평균 이미지수
