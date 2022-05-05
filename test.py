import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

def svdImageMatrix(om, k):
     U, S, Vt = np.linalg.svd(om)
     print("UUUUUUUUUUUUUUUUUUU")
     print(U[:,1].shape)
     print(U[:,1])
     print("SSSSSSSSSSSSSSSSSSS")
     print(S.shape)
     print(S)
     print("VtVtVtVtVtVtVtVtVt")
     print(Vt.shape)
     print(Vt)
     
     cmping = np.matrix(U[:, :k]) * np.diag(S[:k]) * np.matrix(Vt[:k,:])    

     return cmping

def compressImage(image, k):
     cmpRed = svdImageMatrix(image, k)
     # cmpGreen = svdImageMatrix(greenChannel, k)
     # cmpBlue = svdImageMatrix(blueChannel, k)

     # newImage = np.zeros((image.shape[0], image.shape[1], 3), 'uint8')
     # newImage = np.zeros((image.shape[0], image.shape[1], 1), 'uint8')
     newImage = np.zeros((image.shape[0], image.shape[1], 1))

     newImage[..., 0] = cmpRed
     # newImage[..., 1] = cmpGreen
     # newImage[..., 2] = cmpBlue

     return newImage

path = './faces_test/test01.pgm'
# img = mpimg.imread(path)
img = cv2.imread('./faces_training/face01.pgm',-1)

title = "Original Image"
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
weights = [100, 50, 20, 5]

k=int(sys.argv[1])
newImg = compressImage(img, k)

title = " Image after =  %s" %k
# cv2.imshow('image', newImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.title(title)
# plt.imshow(newImg)
# plt.show()    

cv2.imwrite('result.png', newImg)
# newname = os.path.splitext(path)[0] + '_comp_' + str(k) + '.jpg'
     
# for k in weights:
#      newImg = compressImage(img, k)

#      title = " Image after =  %s" %k
#      plt.title(title)
#      plt.imshow(newImg)
#      plt.show()    

#      newname = os.path.splitext(path)[0] + '_comp_' + str(k) + '.jpg'
     # mpimg.imsave(newname, newImg)