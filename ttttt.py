import cv2
import glob
import numpy as np

def compute_l2_distance(input, compare_img):
    # return np.sqrt(np.sum(np.square(compare_img - input)))
    return np.linalg.norm(input-compare_img)

train_img = cv2.imread("./faces_training/face01.pgm",cv2.IMREAD_GRAYSCALE)
test_img = cv2.imread("./faces_test/test01.pgm",cv2.IMREAD_GRAYSCALE)


print(compute_l2_distance(train_img, test_img))