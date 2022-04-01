import cv2
import numpy as np

## Do not erase or modify any lines already written
## Each noise function should return image with noise

def add_gaussian_noise(image):
    # Use mean of 0, and standard deviation of image itself to generate gaussian noise
    mean = 0
    std = image.std()
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    return image
    
def add_uniform_noise(image):
    # Generate noise of uniform distribution in range [0, standard deviation of image)
    std = image.std()
    noise = np.random.uniform(0, std, image.shape)
    image = image + noise
    return image

def apply_impulse_noise(image):
    # Implement pepper noise so that 20% of the image is noisy
    # noised_img = np.zeros(image.shape, np.uint8)
    noised_img = image
    probability = 0.1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rand = np.random.rand(1,1)[0]
            if rand < probability:
                noised_img[i][j] = 5
            elif rand > 1-probability:
                noised_img[i][j] = 230
    return noised_img


def rms(img1, img2):
    # This function calculates RMS error between two grayscale images. 
    # Two images should have same sizes.
    if (img1.shape[0] != img2.shape[0]) or (img1.shape[1] != img2.shape[1]):
        raise Exception("img1 and img2 should have the same sizes.")

    diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))

    return np.sqrt(np.mean(diff ** 2))


if __name__ == '__main__':
    np.random.seed(0)
    original = cv2.imread('bird.jpg', cv2.IMREAD_GRAYSCALE)
    gaussian = add_gaussian_noise(original.copy())
    print("RMS for Gaussian noise:", rms(original, gaussian))
    cv2.imwrite('gaussian.jpg', gaussian)
    
    uniform = add_uniform_noise(original.copy())
    print("RMS for Uniform noise:", rms(original, uniform))
    cv2.imwrite('uniform.jpg', uniform)
    
    impulse = apply_impulse_noise(original.copy())
    print("RMS for Impulse noise:", rms(original, impulse))
    cv2.imwrite('impulse.jpg', impulse)