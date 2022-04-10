from itertools import chain
import cv2
import numpy as np

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
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)
    kernel_size, sigma_s, sigma_r = 7,0.89,700
    print("kernel_size, sigma_s, sigma_r: ", kernel_size, sigma_s, sigma_r)
    # result_img = apply_median_filter(noisy_img, 3)
    # result_img = apply_bilateral_filter(noisy_img, kernel_size, sigma_s, sigma_r)
    result_img = apply_my_filter(noisy_img, 3, 1.3)
    
    # do noise removal
    print("result_img shape: ", result_img.shape)
    print("clean_img shape: ", clean_img.shape)
    # print(calculate_rms(clean_img,result_img))
    cv2.imwrite(dst_path, result_img)
    # result_img = cv2.imread(dst_path)
    print(calculate_rms(clean_img,result_img))
    pass


"""
    You should implement median filter using convolution in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is an int value, which determines kernel size of median filter.

    You should return result image.
    """
def apply_median_filter(img, kernel_size):
    
    img = np.lib.pad(img, [(int(kernel_size/2),int(kernel_size/2)),(int(kernel_size/2),int(kernel_size/2)),(0,0)],"mean")
    img_shape = img.shape
    return_shape = tuple( np.int64(
        (np.array(img_shape) - np.array((kernel_size,kernel_size,0))) + 1
    ))
    height, width, rgb = return_shape
    return_img = np.zeros((height,width,3))
    for z in range(3):
        for x in range(0, height):
            for y in range(0, width):
                temp = img[x:x+kernel_size, y:y+kernel_size,z].ravel()
                temp = np.sort(temp)
                return_img[x,y,z] = temp[int(kernel_size*kernel_size/2)]
    return_img = np.clip(return_img, 0, 255).astype(np.uint8)
    return return_img

"""
    You should implement bilateral filter using convolution in this function.
    It takes at least 4 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s(gaussian function for space)
    'sigma_r' is a int value, which is a sigma value for G_r(gaussian function for range)

    You should return result image.
    """
def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    height, width, _ = img.shape
    img_filterd = np.zeros([height, width, 3])
    kernel_center = kernel_size//2
    spatial_filter = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            spatial_filter[i][j] = np.exp(-((i-kernel_center)*(i-kernel_center)+(j-kernel_center)*(j-kernel_center))/(2*sigma_s*sigma_s))

    for i in range(height):
        for j in range(width):
            total_filter = np.zeros((kernel_size, kernel_size))
            for x in range(kernel_size):
                for y in range(kernel_size):
                    if i-kernel_center+x >= 0 and i-kernel_center+x < height and j-kernel_center+y >= 0 and j-kernel_center+y < width:
                        total_filter[x][y] = spatial_filter[x][y] * np.exp(-np.square(np.linalg.norm(img[i][j]-img[i-kernel_center+x][j-kernel_center+y]))/(2*sigma_r*sigma_r))
            total_weight = total_filter.sum()
            total_filter = total_filter / total_weight
            for x in range(kernel_size):
                for y in range(kernel_size):
                    for rgb in range(3):
                        if i-kernel_center+x >= 0 and i-kernel_center+x < height and j-kernel_center+y >= 0 and j-kernel_center+y < width:
                            img_filterd[i][j][rgb] = img_filterd[i][j][rgb] + img[i-kernel_center+x][j-kernel_center+y][rgb]*total_filter[x][y]
    img_filterd = np.clip(img_filterd, 0, 255).astype(np.uint8)
    return np.asarray(img_filterd)

"""
    You should implement additional filter using convolution.
    You can use any filters for this function, except median, bilateral filter.
    You can add more arguments for this function if you need.

    You should return result image.
    """
def apply_my_filter(img, kernel_size, sigma):
    
    #gaussian filter
    height, width, _ = img.shape
    img_filterd = np.zeros([height, width, 3])
    kernel_center = kernel_size//2
    spatial_filter = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            spatial_filter[i][j] = np.exp(-((i-kernel_center)*(i-kernel_center)+(j-kernel_center)*(j-kernel_center))/(2*sigma*sigma))
    spatial_filter /= (2 * np.pi * sigma * sigma)
    spatial_filter /= spatial_filter.sum()
    for i in range(height):
        for j in range(width):
            for x in range(kernel_size):
                for y in range(kernel_size):
                    for rgb in range(3):
                        if i-kernel_center+x >= 0 and i-kernel_center+x < height and j-kernel_center+y >= 0 and j-kernel_center+y < width:
                            img_filterd[i][j][rgb] = img_filterd[i][j][rgb] + img[i-kernel_center+x][j-kernel_center+y][rgb]*spatial_filter[x][y]
    img_filterd = np.clip(img_filterd, 0, 255).astype(np.uint8)
    return np.asarray(img_filterd)


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have same sizes.")

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int64) - img2.astype(dtype=np.int64))
    return np.sqrt(np.mean(diff ** 2))

if __name__ == "__main__":
    # task1_2("./test_images/fox_noisy.jpg", "./test_images/fox_clean.jpg","./task1_2_result/fox_bilater.jpg")
    # task1_2("./test_images/snowman_noisy.jpg", "./test_images/snowman_clean.jpg","./task1_2_result/snowman_bilater.jpg")
    task1_2("./test_images/snowman_noisy.jpg", "./test_images/snowman_clean.jpg","./task1_2_result/snowman_my.jpg")
    # img1 = cv2.imread('./test_images/fox_clean.jpg')
    # img2 = cv2.imread('./test_images/fox_noisy.jpg')
    # img3 = cv2.imread('./temp_bia.jpg')
    # img3 = cv2.imread('./task1_2_result/fox_bilater.jpg')
    # img3 = cv2.imread('./csdn_3.jpg')

    # print(img1.shape)
    # print(img2.shape)
    # print(img3.shape)
    # print(calculate_rms(img1,img3))