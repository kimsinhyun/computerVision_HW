import cv2
import numpy as np
from sympy import re

def task1_2(src_path, clean_path, dst_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_path' is path for source image.
    'clean_path' is path for clean image.
    'dst_path' is path for output image, where your result image should be saved.

    You should load image in 'src_path', and then perform task 1-2,
    and then save your result image to 'dst_path'.
    """
    # noisy_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)
    # result_img = None
    # result_img = apply_median_filter(noisy_img,3)
    result_img = apply_bilateral_filter(noisy_img,3, 1000,1000)

    # do noise removal

    cv2.imwrite(dst_path, result_img)
    pass


def apply_median_filter(img, kernel_size):
    """
    You should implement median filter using convolution in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is an int value, which determines kernel size of median filter.

    You should return result image.
    """
    #mean is best padding solution = 7.88
    img = np.lib.pad(img, [(int(kernel_size/2),int(kernel_size/2)),(int(kernel_size/2),int(kernel_size/2)),(0,0)],"mean")
    kernel_size = (kernel_size,kernel_size,0)
    img_shape = img.shape
    return_shape = tuple( np.int64(
        (np.array(img_shape) - np.array(kernel_size)) + 1
    ))
    print("return_shape: ", return_shape)
    return_img = np.zeros(return_shape)
    for z in range(3):
        for x in range(0, return_shape[0]):
            for y in range(0, return_shape[1]):
                temp = img[x:x+kernel_size[0], y:y+kernel_size[1],z].ravel()
                temp = np.sort(temp)
                return_img[x,y,z] = temp[int(kernel_size[0]*kernel_size[1]/2)]
    return return_img


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement bilateral filter using convolution in this function.
    It takes at least 4 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s(gaussian function for space)
    'sigma_r' is a int value, which is a sigma value for G_r(gaussian function for range)

    You should return result image.
    """
    print("raw img_shape:", img.shape)
    img = np.lib.pad(img, [(int(kernel_size/2),int(kernel_size/2)),(int(kernel_size/2),int(kernel_size/2)),(0,0)],"mean")
    kernel_size = (kernel_size,kernel_size,0)
    img_shape = img.shape
    print("img_shape:", img_shape)
    return_shape = tuple( np.int64(
        (np.array(img_shape) - np.array(kernel_size)) + 1
    ))
    return_img = np.zeros(return_shape)
    print("return_shape:", return_shape)
    # return_img = img.copy()
    print("return_img.shape:", return_img.shape)
    for i in range(kernel_size[0], img.shape[0] - kernel_size[0]):
        for j in range(kernel_size[0], img.shape[1] - kernel_size[0]):
            for k in range(img.shape[2]):
                weight_sum = 0.0
                pixel_sum = 0.0
                for x in range(-kernel_size[0], kernel_size[0]+1):
                    for y in range(-kernel_size[0], kernel_size[0]+1):
                        spatial_weight = -(x ** 2 + y ** 2) / (2 * (sigma_s ** 2))
                        color_weight = -(int(img[i][j][k]) - int(img[i + x][j + y][k])) ** 2 / (2 * (sigma_r ** 2))
                        weight = np.exp(spatial_weight + color_weight)
                        weight_sum += weight
                        pixel_sum += (weight * img[i + x][j + y][k])
                value = pixel_sum / weight_sum
                return_img[i][j][k] = value
    return return_img.astype(np.uint8)
def apply_my_filter(img):
    """
    You should implement additional filter using convolution.
    You can use any filters for this function, except median, bilateral filter.
    You can add more arguments for this function if you need.

    You should return result image.
    """
    return img


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have same sizes.")

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int) - img2.astype(dtype=np.int))
    return np.sqrt(np.mean(diff ** 2))


if __name__ == "__main__":
    # task1_2("./test_images/cat_noisy.jpg", "./test_images/cat_clean.jpg","./test.jpg")
    task1_2("./test_images/fox_noisy.jpg", "./test_images/fox_clean.jpg","./test.jpg")
    # img1 = cv2.imread('./test_images/fox_clean.jpg')
    # img2 = cv2.imread('./test_images/fox_noisy.jpg')
    # img3 = cv2.imread('./snowman_bilateral_filter.jpg')
    img1 = cv2.imread('./test_images/fox_clean.jpg')
    img2 = cv2.imread('./test_images/fox_noisy.jpg')
    img3 = cv2.imread('./test.jpg')
    print(img1.shape)
    print(img2.shape)
    print(img3.shape)
    print(calculate_rms(img1,img3))