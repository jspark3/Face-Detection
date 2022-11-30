import cv2
import numpy as np
from random import randrange
import time

#################################
####   IMG PROCESSING CODE   ####
#################################

# This loads some pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

##### We will choose an image to detect faces in ######
# img = cv2.imread('Queen.png')
##### Case 2 with multiple faces #####
img = cv2.imread('Multi.png')
# img = cv2.imread('Multi_2.png')

# Convert image to grayscale
gry_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#################################
#### FLATTENING METHOD 1 SVD ####
#################################

# # calculate svd
# U, S, VT = np.linalg.svd(gry_img, full_matrices=False)
# # extracting diagonal Singular Values from the image
# S = np.diag(S)
#
# # Set r value
# # From r = 22, the accuracy goes down.
# r = 25
# # approximation
# reduced_img = U[:, :r]@S[0:r, :r]@VT[:r, :]
# ########## Figure out how long this takes ##########

#######################################################
#### FLATTENING METHOD 2 MANUAL HORIZON FLATTENING ####
#######################################################



####################################################
#### FLATTENING METHOD 3 FULL-VOLUME FLATTENING ####
####################################################


#################################
####   FACE DETECTION CODE   ####
#################################

# Get start time
st = time.time()

# Convert type of image to uint8
# reduced_img = np.array(reduced_img, dtype='uint8')
reduced_img = np.array(gry_img, dtype='uint8')

# Detect Face => [[382 88 172 172]] = [[x, y, w, h]] -> Can detect multiple
face_location = trained_face_data.detectMultiScale(reduced_img)

# Get the end time
et = time.time()
elapsed_time = et - st

# Using coordinates above draw a box around the face on the colored image
for (x, y, w, h) in face_location:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

# Show image
cv2.imshow('Clever Face Detector', img)
# Wait until image window closed
cv2.waitKey()

##### Size of original image in bytes #####
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
size = height * width
size = size * channels

##### Size comparison of SVD #####
# size_svd = reduced_img.shape[0]

print(size)

##### Print size of SVD #####
# print(size_svd)

print('Execution time:', elapsed_time, 'seconds')
print("Code Complete")
