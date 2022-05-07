from enum import EnumMeta
import os
import re
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt




def createDir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ('Error: Creating directory. ' +  dir)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_DB(image_path):
    images = []
    img_file_names = [image for image in os.listdir(image_path)]
    img_file_names.sort(key=natural_keys)
    for image_name in img_file_names:
        image = cv2.imread(os.path.join(train_img_dir, image_name),cv2.IMREAD_GRAYSCALE)
        images.append(np.asarray (image , dtype =np. uint8 ))
    return images,img_file_names

def transform_DB (X):
    mat = np.empty((0,X[0].size),dtype =np.uint8)
    for row in X:
        mat = np.vstack(( mat , np.asarray( row ).reshape(1 , -1))) # 1 x r*c 
    return mat

def get_number_of_components_to_preserve_variance(eigenVals, percentage):
    for ii, eigen_value_cumsum in enumerate(np.cumsum(eigenVals) / np.sum(eigenVals)):
        if eigen_value_cumsum > percentage:
            return ii
def svd (X, percentage, num_dim =0):
    [n,d] = X.shape
    if ( num_dim <= 0) or ( num_dim >n):
        num_dim = n
        mean = X.mean( axis =0)
        X = X - mean
    if n>d:
        C = np.dot(X.T,X) # Covariance Matrix
        [ eigenVals , eigenVecs ] = np.linalg.eigh(C)
    else :
        C = np.dot (X,X.T) # Covariance Matrix
        [ eigenVals , eigenVecs ] = np.linalg.eigh(C)
        eigenVecs = np.dot(X.T, eigenVecs )
        for i in range (n):
            eigenVecs [:,i] = eigenVecs [:,i]/ np.linalg.norm( eigenVecs [:,i])
    # sort eigenVecs descending by their eigenvalue
    idx = np.argsort (- eigenVals )
    eigenVals = eigenVals [idx ]
    eigenVecs = eigenVecs [:, idx ]
    print("eigenVals", eigenVals)
    num_dim = get_number_of_components_to_preserve_variance(eigenVals, percentage)
    # select only num_dim
    eigenVals = eigenVals [0: num_dim ].copy ()
    eigenVecs = eigenVecs [: ,0: num_dim ].copy ()
    return eigenVals , eigenVecs , mean , num_dim  

def project (W , X , mu):
    return np.dot (X - mu , W)
def reconstruct (W , Y , mu) :
    return np.dot (Y , W.T) + mu
# def reconstruct(img_idx, e_faces, weights, mu, npcs):
# 	# dot weights with the eigenfaces and add to mean
# 	recon = mu + np.dot(weights[img_idx, 0:npcs], e_faces[:, 0:npcs].T)
# 	return recon
def Euclidean_distance(p,q):
    return np.linalg.norm(q-p)
def predict (W, mu , projections, y, X):
    minDist = float("inf")
    minClass = -1
    Q = project (W, X.reshape (1 , -1) , mu)
    for i in range (len(projections)):
#         dist = dist_metric( projections[i], Q)
        dist = Euclidean_distance( Q, projections[i])
        if dist < minDist:
            minDist = dist
            minClass = i
    return minClass

def output_step1(output, percentage, num_dim):
    output.write("########## STEP 1 ##########\n")
    output.write(f"Input Percentage: {percentage}\n")
    output.write(f"Selected Dimension: {num_dim}\n")
    output.write("\n")
    return output

def return_reconstructed_img(num,images,eigenVecs, num_dim, mean):
    projected = project(eigenVecs[:,num:num_dim], images[num].reshape(1,-1), mean)
    reconstructed = reconstruct(eigenVecs[:,num:num_dim], projected ,mean)
    reconstructed = reconstructed.reshape(images[num].shape)
    return reconstructed

if __name__ == "__main__":
    student_id = "2016147588"
    createDir(student_id)
    percentage=float(sys.argv[1])
    train_img_dir = 'faces_training'
    """
    1. train 이미지들을 모두 읽어와서 하나의 데이터 베이스로 저장
    """
    images, images_names = create_DB(train_img_dir)

    """
    2. transform_DB: N*N행렬을 N^2 * 1행렬로 Flat, 각 Row가 하나의 image이다.
    3. 
    """
    eigenVals, eigenVecs, mean, num_dim = svd(transform_DB(images), percentage)
    
    # plt.imshow(images[1], cmap=plt.cm.gray)
    # plt.title("Mean Face")
    # plt.show()
    """
    4. save reconstruceted images
    """
    for i, file_name in enumerate(images_names):
        file_path = "./2016147588/" + file_name
        reconstructed = return_reconstructed_img(i, images, eigenVecs, num_dim, mean)
        cv2.imwrite(file_path,reconstructed)
    # for i, image in enumerate()

    projections = []
    for xi in images:
        projections.append(project(eigenVecs, xi.reshape(1 , -1), mean))

    temp_image = cv2.imread('./faces_test/test01.pgm',cv2.IMREAD_GRAYSCALE)
    test_image = np. asarray (temp_image , dtype =np.uint8 )
    predicted = predict(eigenVecs, mean , projections, images_names, test_image)
    plt.imshow(images[predicted], cmap=plt.cm.gray)
    plt.title("Mean Face")
    plt.show()

    output = open('./2016147588/output.txt', 'w')
    output_step1(output,percentage, num_dim)
    

