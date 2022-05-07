import os
import re
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
#이미지 출력 함수
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def show_img(temp):
    cv2.imshow('image', temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def compute_MSE(raw_img, result):
    return np.mean(np.square(raw_img - result))

def compute_l2_distance(input, compare_img):
    return np.linalg.norm(input-compare_img)

def svd_pca(raw_img, rate):
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
    # mse = compute_MSE(raw_img, result)
    # 결과 저장
    # save_img(student_id,image_path,result)
    return result, K

def step1_output(rate, K):
    output = open('./2016147588/output.txt', 'w')
    output.write("########## STEP 1 ##########\n")
    output.write(f"Input Percentage: {rate}\n")
    output.write(f"Selected Dimension: {K}\n")
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

def split_image_and_save_into_dir(imgs,train_img_file_list):
    zipped_images_list = []
    for i,file in enumerate(train_img_file_list):
        img = imgs[:,168*i:168*(i+1)]
        file_path = "./2016147588/" + file
        cv2.imwrite(file_path,img)
        zipped_images_list.append(img)
    return zipped_images_list

#main func
if __name__ == "__main__":
    student_id = "2016147588"
    createDir(student_id)
    rate=float(sys.argv[1])
    #image names
    train_img_file_list = os.listdir("faces_training") # dir is your directory path
    test_img_file_list = os.listdir("faces_test")
    
    #sort filenames by number inside
    train_img_file_list.sort(key=natural_keys)
    test_img_file_list.sort(key=natural_keys)

    #real image objects
    train_img_obj_list = []
    test_img_obj_list = []

    mse_list = []
    recognition_result_list = []

    #read train images =================step 1, 2=================
    for img_name in train_img_file_list:
        # print(img_name)
        image_path = f"./faces_training/{img_name}"
        raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        train_img_obj_list.append(raw_img)
        #run svd and caculate mse


    # =================step 1=================
    train_img_concat = np.concatenate(train_img_obj_list, axis=1)
    zipped_imgs ,K = svd_pca(train_img_concat, rate) #apply svd and mse
    plt.imshow(zipped_imgs, 'gray')
    plt.show()
    # show_img(zipped_imgs)
    print(zipped_imgs.shape)
    print(K)
    zipped_train_images_obj_list = split_image_and_save_into_dir(zipped_imgs, train_img_file_list)

    # =================step 2=================
    for i in range(len(zipped_train_images_obj_list)):
        temp_mse = compute_MSE(zipped_train_images_obj_list[i],train_img_obj_list[i])
        mse_list.append(temp_mse)
    


    #read test images =================step 3=================
    for img in test_img_file_list:
        # print(img)
        image_path = f"./faces_test/{img}"
        test_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        test_img_obj_list.append(test_img)

    for i, test_img in enumerate(test_img_obj_list):
        temp_distance_list = []
        for j, train_img in enumerate(train_img_obj_list):
            # print(train_img_file_list[j])
            temp_distance = compute_l2_distance(train_img, test_img)
            temp_distance_list.append(temp_distance)
        print(test_img_file_list[i])
        recognition_result = train_img_file_list[np.argmin(temp_distance_list)]
        print(np.round(temp_distance_list,2))
        print(np.min(temp_distance_list))
        print(np.argmin(temp_distance_list))
        recognition_result_list.append(recognition_result)
    print("train_img_file_list")
    print(train_img_file_list)
    print("test_img_file_list")
    print(test_img_file_list)
    print("recognition_result_list")
    print(recognition_result_list)
    print("min temp_distance_list")
    print(recognition_result)
    output = step1_output(rate, K)
    output = step2_output(output, mse_list)
    output = step3_output(output, recognition_result_list)

    output.close()
    print(compute_l2_distance(train_img_obj_list[0],test_img_obj_list[0]))










