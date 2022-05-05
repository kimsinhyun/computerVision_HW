# import os
# import cv2
# import numpy as np
# import sys

# def compute_l2_distance(input, compare_img):
#     # return np.sqrt(np.sum(np.square(compare_img - input)))
#     return np.linalg.norm(input-compare_img)
# def atoi(text):
#     return int(text) if text.isdigit() else text

# def natural_keys(text):
#     return [ atoi(c) for c in re.split(r'(\d+)', text) ]


# if __name__ == "__main__":
#      test_img = cv2.imread("./faces_test/test01.pgm", cv2.IMREAD_GRAYSCALE)

#      train_img = cv2.imread("./faces_training/face01.pgm", cv2.IMREAD_GRAYSCALE)

#      train_img_list = os.listdir("faces_training") # dir is your directory path
     
#      #sort filenames by number inside
#      train_img_list.sort(key=natural_keys)
#      test_img_list.sort(key=natural_keys)

#      print("train_img_list")
#      print(train_img_list)
#      print("test_img_list")
#      print(test_img_list)

#      #real image objects
#      train_img_obj_list = []
#      test_img_obj_list = []

#      K_list = []
#      mse_list = []
#      recognition_result_list = []

#      #for step 1 & 2
#      for img_name in train_img_list:
#           image_path = f"./faces_training/{img_name}"
#           raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#           train_img_obj_list.append(raw_img)

#      #read test images
#      for img in test_img_list:
#           image_path = f"./faces_test/{img}"
#           test_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#           test_img_obj_list.append(test_img)

#      for test_img in test_img_obj_list:
#           temp_distance_list = []
#           for train_img in train_img_obj_list:
#                temp_distance = compute_l2_distance(test_img,train_img)
#                temp_distance_list.append(temp_distance)
#           recognition_result = train_img_list[np.argmin(temp_distance_list)]
#           print(np.min(temp_distance_list))
#           recognition_result_list.append(recognition_result)
