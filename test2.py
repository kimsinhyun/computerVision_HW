import os
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
    file_name = "./" + student_id + "/" + img_file_name
    fout = open(file_name, 'wb')
    pmg_header = "P5" +  "\n" + str(img.shape[0]) + " " + str(img.shape[1]) + "\n" + str(255) + "\n"
    pgm_header_byte = bytearray(pmg_header, 'utf-8')
    fout.write(pgm_header_byte)
    img = np.reshape(img,(img.shape[0],img.shape[1],1))

    for i in range(img.shape[1]):
        bnd = list(img[i,:])
        fout.write(bytearray(bnd)) # for 8-bit data only
    fout.close()
    # cv2.imwrite(file_name, dst)
    # print(path)

#min max scaler
def min_max_scaler(img):
    for i in range(3):
        MAX = np.max(img[:, :, i])
        MIN = np.min(img[:, :, i])
        img[:, :, i] = (img[:, :, i] - MIN) / (MAX - MIN)  
    return img

#정보 출력 함수
def print_info(image_path, rate, K, origin_image, u_shape, s_shape, vT_shape, zip_rate):
    print("원본 이미지:\t\t", image_path)
    print("보존률:\t\t", rate)
    print("사용된 dimension:\t\t", K)
    print("원본 이미지 크기：\t\t", origin_image.shape)
    print("차원축소 후 각 matrix 크기:\t\t3 x ({} + {} + {})". format(u_shape, s_shape, vT_shape))
    print("압축율：\t\t", zip_rate)

def zip_image_by_svd(student_id, image_path, origin_image, rate = 0.8):
    result = np.zeros(origin_image.shape)
    u_shape, s_shape, vT_shape  = 0 , 0 , 0
    for chan in range(origin_image.shape[2]):
        #각 체널마다 svd 특이값 추출, sigma 안에 원본 데이터의 얼만큼의 표현율을 담는지가 들어있음
        U, sigma, V = np.linalg.svd(origin_image[:, :, chan])
        K, temp = 0, 0
        # 보존률을 만족하기 위해 필요한 singular value 수 (K)
        while (temp / np.sum(sigma)) < rate:
            temp += sigma[K]
            K += 1
        # 특이값 matrix를 위한 행렬
        S = np.diag(sigma)[:K,:K]
        # 위에서 얻어진 S 행렬을 통해 축소된 K 차원에 project 
        result[:, :, chan] = (U[:, 0:K].dot(S)).dot(V[0:K, :])
        u_shape = U[:, 0:K].shape
        s_shape = S.shape
        vT_shape = V[0:K, :].shape
    #scaling 적용(min max scaler 를 사용함)
    result = min_max_scaler(result)
    #결과를 0~255의 int 배열로 저장 (이미지이기 때문에)
    result  = np.round(result * 255).astype('int')
    # 결과 저장
    
    save_img(student_id,image_path,result)
    # 압축률 계산
    zip_rate =(origin_image.size -3 * (u_shape[0] * u_shape[1] + s_shape[0] * s_shape[1] + vT_shape[0] * vT_shape[1])) / (origin_image.size)
    #정보 표시
    # print_info(image_path, rate, K, origin_image, u_shape, s_shape, vT_shape, zip_rate)
    

def main(student_id, image_path, rate):
    #read image
    origin_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    #use numpy svd to reduce dimension 
    zip_image_by_svd(student_id, image_path, origin_image, rate)
    
#main func
if __name__ == "__main__":
    student_id = "2016147588"
    createDir(student_id)
    rate=float(sys.argv[1])
    image_path = './faces_training/face01.pgm'
    main(student_id, image_path, rate)