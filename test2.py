import os
import re
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
#이미지 출력 함수
def show_img(title,img):
    plt.figure(figsize= (12, 12))
    plt.title(title)
    plt.imshow(img)
    plt.show()

#directory 생성
def createDir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ('Error: Creating directory. ' +  dir)

def save_img(student_id, image_path,img):
    img_file_name = image_path.split("/")[2]
    file_path = "./" + student_id + "/" + img_file_name
    cv2.imwrite(file_path,img)

#min max scaler
def min_max_scaler(img):
    temp_max = np.max(img[:, :,])
    temp_min = np.min(img[:, :])
    img[:, :] = (img[:, :] - temp_min) / (temp_max - temp_min)  
    return img

#정보 출력 함수
def print_info(image_path, rate, K, raw_img, u_shape, s_shape, vT_shape):
    print("원본 이미지:\t\t\t", image_path)
    print("보존률:\t\t\t\t", rate)
    print("사용된 dimension:\t\t", K)
    print("원본 이미지 크기：\t\t", raw_img.shape)
    print("차원축소 후 각 matrix 크기:\t3 x ({} + {} + {})". format(u_shape, s_shape, vT_shape))

def compute_MSE(raw_img, result):
    return np.mean(np.square(raw_img - result))

def svd_pca(student_id, image_path, raw_img, rate = 0.8):
    result = np.zeros(raw_img.shape)
    u_shape, s_shape, vT_shape  = 0 , 0 , 0
    U, sigma, V = np.linalg.svd(raw_img[:, :])
    K, temp = 0, 0
    # 보존률을 만족하기 위해 필요한 singular value 수 (K)
    while (temp / np.sum(sigma)) < rate:
        temp += sigma[K]
        K += 1
    # 특이값 matrix를 위한 diagnoal행렬
    S = np.diag(sigma)[:K,:K]
    # 위에서 얻어진 S 행렬을 통해 축소된 K 차원에 project 
    result[:, :] = (U[:, 0:K].dot(S)).dot(V[0:K, :])
    u_shape = U[:, 0:K].shape
    s_shape = S.shape
    vT_shape = V[0:K, :].shape
    #scaling 적용(min max scaler 를 사용함)
    result = min_max_scaler(result)
    #결과를 0~255의 int 배열로 저장 (이미지이기 때문에)
    result  = np.round(result * 255).astype('int')
    #compute MSE
    mse = compute_MSE(raw_img, result)
    # 결과 저장
    save_img(student_id,image_path,result)
    # 압축률 계산
    #정보 표시
    # print_info(image_path, rate, K, raw_img, u_shape, s_shape, vT_shape)
    return K, mse

def write_output_img(rate, K_list, mse_list):
    output = open('./2016147588/output.txt', 'w')
    output.write("########## STEP 1 ##########\n")
    output.write(f"Input Percentage: {rate}\n")
    output.write(f"Input Percentage: {rate}\n")


#main func
if __name__ == "__main__":
    student_id = "2016147588"
    createDir(student_id)
    rate=float(sys.argv[1])
    img_list = os.listdir("faces_training") # dir is your directory path
    
    img_obj_list = []
    K_list = []
    mse_list = []
    for img_name in img_list:
        image_path = f"./faces_training/{img_name}"
        raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_obj_list.append(raw_img)
        temp_K, temp_mse = svd_pca(student_id, image_path, raw_img, rate)
        K_list.append(temp_K)
        mse_list.append(temp_mse)
    print(K_list)
    print(mse_list)

    write_output_img(rate, K_list, mse_list)