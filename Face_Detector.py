import cv2

#This loads some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_defalut.xml')

# We will choose an image to detect faces in
img = cv2.imread('Face.JPG')



print("Code Complete")
