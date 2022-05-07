import glob
from statistics import variance
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_images(directory):
	# get a list of train images file name
    img_file_names = glob.glob(directory + '/*.pgm')
    img_file_names.sort(key=natural_keys)
    # load a greyscale of each image
    imgs = np.array([cv2.imread(i, cv2.IMREAD_GRAYSCALE).flatten() for i in img_file_names])
    return imgs, img_file_names, len(img_file_names)

def choose_raw_img_from_concated_img(idx,data):
    return data[idx*192:(idx+1)*192,:]

def pca(data):
    mu = np.mean(data, 0)
    # mean adjust the data
    ma_data = data - mu
    #execute svd
    e_faces, sigma, v = np.linalg.svd(ma_data.transpose(), full_matrices=False)
    weights = np.dot(ma_data, e_faces)
    return e_faces, sigma, weights, mu

def get_number_of_components_to_preserve_variance(sigma, percentage):
    for ii, eigen_value_cumsum in enumerate(np.cumsum(sigma) / np.sum(sigma)):
        if eigen_value_cumsum > percentage:
            return ii

def reconstruct(img_idx, e_faces, weights, mu, npcs):
	reconstructed = mu + np.dot(weights[img_idx, 0:npcs], e_faces[:, 0:npcs].T)
	return reconstructed

def compute_MSE(raw_img, reconstruced):
    return np.mean(np.square(raw_img - reconstruced))

def show_img(data):
    plt.imshow(data, cmap=plt.cm.gray)
    plt.title("Mean Face")
    plt.show()

def Euclidean_distance(p,q):
    return np.linalg.norm(p-q)

def normalization(img):
    norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm

def predict(reconstructed_imgs, test_img, data_file_names):
    distance_list = []
    for reconstructed in reconstructed_imgs:
        distance = Euclidean_distance(test_img,reconstructed)
        distance_list.append(distance)
    closest_img_idx = np.argmin(distance_list)
    return data_file_names[closest_img_idx]

def step1_output(out ,variance_percentage, n_dim):
    output.write("########## STEP 1 ##########\n")
    output.write(f"Input Percentage: {variance_percentage}\n")
    output.write(f"Selected Dimension: {n_dim}\n")
    output.write("\n")
    return output
    
def step2_output(output, mse_list):
    output.write("########## STEP 2 ##########\n")
    output.write("Reconstruction error\n")
    output.write(f"Average: {round(np.mean(mse_list),4)}\n")
    for i, mse in enumerate(mse_list):
        output.write(f"{str(i+1).zfill(2)}: {round(mse,4)}\n")
    output.write("\n")
    return output

def step3_output(output, recognition_result_list):
    output.write("########## STEP 3 ##########\n")
    for i, recog_result in enumerate(recognition_result_list):
        output.write(f"test{str(i+1).zfill(2)}.pgm ==> {recog_result}\n")
    return output
if __name__ == "__main__":
    variance_percentage = float(sys.argv[1])
    in_dir  = "faces_training"
    out_dir = "2016147588/"
    test_dir = "faces_test"
    img_dims = (192, 168)

    #load train datas
    data, data_file_names,data_num = load_images(in_dir)
    raw_data = data.reshape(192*data_num, 168)
    #compute train datas pca(svd)
    e_faces, sigma, weights, mu = pca(data)

    #test train datas
    compressed_test_data, test_data_file_names,test_data_num = load_images(test_dir)
    test_data = compressed_test_data.reshape(192*test_data_num, 168)


    #save images
    reconstructed_imgs = []
    for p in range(data.shape[0]):
        n_dim = get_number_of_components_to_preserve_variance(sigma,variance_percentage)
        reconstructed = reconstruct(p, e_faces, weights, mu, n_dim).reshape(img_dims) 
        file_name = out_dir + "face" + f"{str(p+1).zfill(2)}.pgm"
        reconstructed_imgs.append(reconstructed)
        cv2.imwrite(file_name, reconstructed)
    mse_list = []
    #caculate reconstructed images mse
    for i, reconstructed in enumerate(reconstructed_imgs):
        # show_img(reconstructed)
        mse = compute_MSE(reconstructed,choose_raw_img_from_concated_img(i,raw_data))
        mse_list.append(mse)
        # show_img(reconstructed)
        # show_img(raw_data[i*192:(i+1)*192,:])
    
    # show_img(test_data)

    predict_list = []
    for i in range(test_data_num):
        predict_result = predict(reconstructed_imgs, choose_raw_img_from_concated_img(i,test_data), data_file_names)
        predict_list.append(predict_result)

    output = open('./2016147588/output.txt', 'w')
    step1_output(output, variance_percentage, n_dim)
    step2_output(output, mse_list)
    step3_output(output,predict_list)
    output.close()
