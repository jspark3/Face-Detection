import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

# This loads some pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_defalut.xml')

# We will choose an image to detect faces in
# img = cv2.imread('Queen.png')
img = cv2.imread('Multi.png')
# Convert image to grayscale
gry_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

U, S, VT = np.linalg.svd(gry_img, full_matrices=False)
S = np.diag(S)
# extracting diagnal Singular Values from the image

reduced_img = plt.imshow(U[:,:20]@S[0:20,:20]@VT[:20,:])

# Detect Face => [[382 88 172 172]] = [[x, y, w, h]] -> Can detect multiple
face_location = trained_face_data.detectMultiScale(reduced_img)

# Draw rectangle around face
for (x, y, w, h) in face_location:
    cv2.rectangle(reduced_img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

# Show image
cv2.imshow('Clever Face Detector', reduced_img)
# Wait until image window closed
cv2.waitKey()

print("Code Complete")
