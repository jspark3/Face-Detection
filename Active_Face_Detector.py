import cv2
from random import randrange

# This loads some pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_defalut.xml')

# To capture video from webcam.
webcam = cv2.VideoCapture(0)

# Iterate over frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()
    # Convert image to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Face => [[382 88 172 172]] = [[x, y, w, h]] -> Can detect multiple
    face_location = trained_face_data.detectMultiScale(gray_img)

    # Draw rectangle around face
    for (x, y, w, h) in face_location:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Show image
    cv2.imshow('Clever Face Detector', frame)
    # Wait until image window closed
    cv2.waitKey(1)
