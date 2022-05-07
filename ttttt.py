import glob
import numpy as np
import os
import pdb
import cv2
import matplotlib.pyplot as plt
import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# loads a greyscale version of every jpg image in the directory.
# INPUT  : directory
# OUTPUT : imgs - n x p array (n images with p pixels each)
def load_images(directory):
	# get a list of all the picture filenames
    jpgs = glob.glob(directory + '/*.pgm')
    jpgs.sort(key=natural_keys)
    print(jpgs)
    # load a greyscale version of each image
    imgs = np.array([cv2.imread(i, cv2.IMREAD_GRAYSCALE).flatten() for i in jpgs])
    return imgs

# chooses the first image from each folder in indir,
# and saves a copy of the images in the outdir.
# INPUT  : indir  - directory to retrieve folders from
#          outdir - directory to save images to
def choose_images(indir, outdir):
	counter = 0
	for folder in glob.glob(indir + '/*'):
		filenames = glob.glob(folder + '/*.jpg')
		# just choose the first image
		img = cv2.imread(filenames[0], cv2.IMREAD_GRAYSCALE)
		cv2.imwrite(outdir + "/img_" + str(counter) + ".jpg", img)
		counter = counter + 1

# Run Principal Component Analysis on the input data.
# INPUT  : data    - an n x p matrix
# OUTPUT : e_faces -
#          weights -
#          mu      -
def pca(data):
    mu = np.mean(data, 0)
    print(data)
    # mean adjust the data
    ma_data = data - mu
    # run SVD
    e_faces, sigma, v = np.linalg.svd(ma_data.transpose(), full_matrices=False)
    # pdb.set_trace()
    # compute weights for each image
    weights = np.dot(ma_data, e_faces)
    return e_faces, sigma, weights, mu

def get_number_of_components_to_preserve_variance(sigma, percentage):
    for ii, eigen_value_cumsum in enumerate(np.cumsum(sigma) / np.sum(sigma)):
        if eigen_value_cumsum > percentage:
            return ii
    # var_explained = np.round(weight**2/np.sum(weight**2))
    # return var_explained
# reconstruct an image using the given number of principal
# components.
def reconstruct(img_idx, e_faces, weights, mu, npcs):
	# dot weights with the eigenfaces and add to mean
	recon = mu + np.dot(weights[img_idx, 0:npcs], e_faces[:, 0:npcs].T)
	return recon
	
def save_image(out_dir, subdir, img_id, img_dims, data):
	directory = out_dir + "/" + subdir
	if not os.path.exists(directory): os.makedirs(directory)
	cv2.imwrite(directory + "/image_" + str(img_id) + ".jpg", data.reshape(img_dims))

def run_experiment():
    in_dir  = "faces_training"
    out_dir = "2016147588/"
    img_dims = (192, 168)

    data = load_images(in_dir)
    e_faces, sigma, weights, mu = pca(data)

    # save mean photo
    #cv2.imwrite(out_dir + "/mean.jpg", mu.reshape(img_dims))

    # save each eigenface as an image
    for i in range(e_faces.shape[1]):
        continue
        #save_image(out_dir, "eigenfaces", i, e_faces[:,i])

    # reconstruct each face image using an increasing number
    # of principal components

    for p in range(data.shape[0]):
        # for i in range(data.shape[0]):
            # reconstructed.append(reconstruct(p, e_faces, weights, mu, i))
        print("eigenVals")
        print(weights)
        print("get_number_of_components_to_preserve_variance(e_faces,0.9)")
        n_components = get_number_of_components_to_preserve_variance(sigma,0.3)
        print(n_components)
        reconstructed = reconstruct(p, e_faces, weights, mu, n_components).reshape(img_dims) 
        file_name = out_dir + "face" + f"{str(p+1).zfill(2)}.pgm"
        cv2.imwrite(file_name, reconstructed)
    #         img_id = str(i / 10) + "a" + str(i % 10)
    # print(reconstructed[0].reshape(img_dims))
    # plt.imshow(reconstructed[1].reshape(img_dims), 'gray')
    # plt.show()
run_experiment()