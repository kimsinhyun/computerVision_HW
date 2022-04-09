import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####

def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    
    freq_1_1 = img[:img.shape[0]//2, :img.shape[0]//2]
    freq_1_2 = img[img.shape[0]//2:, :img.shape[0]//2]
    freq_2_1 = img[:img.shape[0]//2, img.shape[1]//2:]
    freq_2_2 = img[img.shape[0]//2:, img.shape[1]//2:]
    freq_2 = np.concatenate((freq_2_2, freq_2_1))
    freq_1 = np.concatenate((freq_1_2, freq_1_1))
    freq = np.concatenate((freq_2, freq_1),axis=1)

    # m_spectrum = 20*np.log(np.abs(freq))
    return freq


def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    freq_1_1 = img[:img.shape[0]//2, :img.shape[0]//2]
    freq_1_2 = img[img.shape[0]//2:, :img.shape[0]//2]
    freq_2_1 = img[:img.shape[0]//2, img.shape[1]//2:]
    freq_2_2 = img[img.shape[0]//2:, img.shape[1]//2:]
    freq_2 = np.concatenate((freq_2_2, freq_2_1))
    freq_1 = np.concatenate((freq_1_2, freq_1_1))
    freq = np.concatenate((freq_2, freq_1),axis=1)
    return freq

def fm_spectrum(img):
    '''
    This function should get the frequency magnitude spectrum of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    You may have to multiply the resultant spectrum by a certain magnitude in order to display it correctly.
    '''
    img = np.fft.fft2(img)
    img = fftshift(img)
    m_spectrum = 20*np.log(np.abs(img))
    return m_spectrum

def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''
    height, width = img.shape
    fft = np.fft.fft2(img)
    fftshifted = fftshift(fft)
    #중앙에 검은색 동그라미
    for i in range(height):
        for j in range(width):
            if (i - (height)/2)**2 + (j - (width)/2)**2 >= r**2:
                fftshifted[i, j] = 0
    
    fftshifted = np.fft.ifftshift(fftshifted)
    ifft = np.fft.ifft2(fftshifted)
    
    temp = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            temp[i, j] = ifft[i, j].real
    
    max, min = np.max(temp) , np.min(temp)
    return_img = np.zeros((height, width), dtype = "uint8")
    
    for i in range(height):
        for j in range(width):
            return_img[i, j] = 255*(temp[i, j] - min)/(max - min)
    return return_img

def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''
    height, width = img.shape
    fft = np.fft.fft2(img)
    fftshifted = fftshift(fft)
    #중앙에 검은색 동그라미
    for i in range(height):
        for j in range(width):
            if (i - (height)/2)**2 + (j - (width)/2)**2 <= r**2:
                fftshifted[i, j] = 1
            
    fftshifted = np.fft.ifftshift(fftshifted)
    ifft = np.fft.ifft2(fftshifted)
    
    temp = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            temp[i, j] = ifft[i, j].real
    
    # max, min = np.max(temp) , np.min(temp)
    # # return_img = np.zeros((height, width), dtype = "uint8")
    # return_img = np.zeros((height, width))
    
    # for i in range(height):
    #     for j in range(width):
    #         return_img[i, j] = 255*(temp[i, j].real - min)/(max - min)

    return temp

def denoise1(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''

    height, width = img.shape
    fft = np.fft.fft2(img)
    fftshifted = fftshift(fft)
    r1 = 20
    #중앙에 검은색 동그라미
    #특정 구역을 1 시그마 값으로 outlier를 찾고 옆에 값으로 치환
    test_2 = [fftshifted.mean() - 1 * fftshifted.std(), fftshifted.mean() + 1 * fftshifted.std()]
    for i in range(height):
        for j in range(width):
            # if ((i - (height)/2)**2 + (j - (width)/2)**2 >= r1**2 and (i - (height)/2)**2 + (j - (width)/2)**2 <= r2**2):
            if (i - (height)/2)**2 + (j - (width)/2)**2 >= r1**2:
                if(np.abs(i - (height//2)) >= 5):
                    if(np.abs(j - (width//2)) >= 5):
                        if(fftshifted[i, j] <= test_2[0]):
                            fftshifted[i, j] = fftshifted[i+10, j+10]
                        elif(fftshifted[i, j]  >= test_2[1]):
                            fftshifted[i, j] = fftshifted[i+10, j+10]
    
    fftshifted = np.fft.ifftshift(fftshifted)
    ifft = np.fft.ifft2(fftshifted)
    
    temp = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            temp[i, j] = ifft[i, j].real
    
    max, min = np.max(temp) , np.min(temp)
    return_img = np.zeros((height, width), dtype = "uint8")
    
    for i in range(height):
        for j in range(width):
            return_img[i, j] = 255*(temp[i, j] - min)/(max - min)
    return return_img

def denoise2(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    height, width = img.shape
    fft = np.fft.fft2(img)
    fftshifted = fftshift(fft)
    r1 = 27
    r2 = 28
    test = np.mean(fftshifted[height//2-14:height//2+14,width//2-14:width//2+14])
    print(height//2-1)
    #중앙에 검은색 동그라미
    for i in range(height):
        for j in range(width):
            if ((i - (height)/2)**2 + (j - (width)/2)**2 >= r1**2 and (i - (height)/2)**2 + (j - (width)/2)**2 <= r2**2):
                if(i<=height//2-2 or i>=height//2+2):
                    if(j<=height//2-1 or j>=height//2+2):
                        fftshifted[i, j] = test
    
    fftshifted = np.fft.ifftshift(fftshifted)
    ifft = np.fft.ifft2(fftshifted)
    
    temp = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            temp[i, j] = ifft[i, j].real
    
    max, min = np.max(temp) , np.min(temp)
    return_img = np.zeros((height, width), dtype = "uint8")
    
    for i in range(height):
        for j in range(width):
            return_img[i, j] = 255*(temp[i, j] - min)/(max - min)
    return return_img


#################

# Extra Credit
def dft2(img):
    '''
    Extra Credit. 
    Implement 2D Discrete Fourier Transform.
    Naive implementation runs in O(N^4).
    '''
    return img

def idft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Discrete Fourier Transform.
    Naive implementation runs in O(N^4). 
    '''
    return img

def fft2(img):
    '''
    Extra Credit. 
    Implement 2D Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

def ifft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

if __name__ == '__main__':
    img = cv2.imread('task2_filtering.png', cv2.IMREAD_GRAYSCALE)
    noised1 = cv2.imread('task2_noised1.png', cv2.IMREAD_GRAYSCALE)
    noised2 = cv2.imread('task2_noised2.png', cv2.IMREAD_GRAYSCALE)

    low_passed = low_pass_filter(img)
    high_passed = high_pass_filter(img)
    denoised1 = denoise1(noised1)
    denoised2 = denoise2(noised2)

    # save the filtered/denoised images
    cv2.imwrite('low_passed.png', low_passed)
    cv2.imwrite('high_passed.png', high_passed)
    cv2.imwrite('denoised1.png', denoised1)
    cv2.imwrite('denoised2.png', denoised2)

    # draw the filtered/denoised images
    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_passed, 'Low-pass')
    drawFigure((2,7,3), high_passed, 'High-pass')
    drawFigure((2,7,4), noised1, 'Noised')
    drawFigure((2,7,5), denoised1, 'Denoised')
    drawFigure((2,7,6), noised2, 'Noised')
    drawFigure((2,7,7), denoised2, 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_passed), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_passed), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(noised1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoised1), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(noised2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoised2), 'Spectrum')

    plt.show()