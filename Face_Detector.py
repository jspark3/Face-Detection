import cv2
from random import randrange

# This loads some pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_defalut.xml')

# We will choose an image to detect faces in
# img = cv2.imread('Queen.png')
img = cv2.imread('Multi.png')
# Convert image to grayscale
gry_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# webcam = cv2.VideoCapture(0)

# Detect Face => [[382 88 172 172]] = [[x, y, w, h]] -> Can detect multiple
face_location = trained_face_data.detectMultiScale(gry_img)

# Draw rectangle around face
for (x, y, w, h) in face_location:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

# Show image
cv2.imshow('Clever Face Detector', img)
# Wait until image window closed
cv2.waitKey()

print("Code Complete")
