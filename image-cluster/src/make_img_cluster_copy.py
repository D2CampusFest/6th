import os
import numpy as np
from config import *

# get list of files
img_paths = os.listdir(IMG_DIR)
img_paths.sort()
img_paths = [filename for filename in img_paths if filename.endswith(IMG_EXT)]

# load predicted labels
labels_pred = np.load(os.path.join(DATA_DIR, LABELS_PRED + ".npy"))

# make result dir
if os.path.exists(CLUSTER_DIR) is False:
    os.makedirs(CLUSTER_DIR)
