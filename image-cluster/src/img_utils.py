"""
두 이미지의 픽셀값이 완전히 일치하는지 비교하는 모듈입니다.
"""
import cv2
import numpy as np


def check_img_dim(img1, img2):
    if np.ndim(img1) != np.ndim(img2):
        return False

    for i in range(np.ndim(img1)):
        if np.shape(img1)[i] != np.shape(img2)[i]:
            return False

    return True


def img_diff_norm(img1, img2):
    # check img dim
    if not check_img_dim(img1, img2):
        return False

    # subtract and compute norm
    diff = cv2.subtract(img1, img2)
    norm = cv2.norm(diff)

    return norm


def is_same_image(img1, img2):
    # return True if diff norm is 0
    if img_diff_norm(img1, img2) > 0:
        return False
    else:
        return True


def find_first_same_image_in_list(img_list, test_img):
    found = -1
    for i in range(len(img_list)):
        if is_same_image(img_list[i], test_img):
            found = i
            break
    return found


def test():
    a = np.array([[[1, 2], [3, 4]]])
    b = np.array([[[1, 2], [3, 4]]])
    c = np.array([[[1, 2], [3, 5]]])

    print(is_same_image(a, b))
    print(is_same_image(a, c))

    print(find_first_same_image_in_list([a, b], c))
    print(find_first_same_image_in_list([a, c], b))


if __name__ == '__main__':
    test()
