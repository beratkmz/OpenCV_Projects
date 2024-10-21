import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Image File 1','Image File 2','Image File 3']

# Loading trained labels and features files
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

capture = cv.VideoCapture(0)

while True:
    # Frame reading
    _,img = capture.read()

    # Converting gray channel (3 BGR channel to 1 GRAY channel)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Face Detection
    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=6)

    # Create rectangle and text for detected face
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(img, str(people[label]),(20,20), cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),2)
        cv.rectangle(img, (x,y),(x+w,y+h),(0,0,255),2)

    cv.imshow('Detected Face',img)

    # Pausing the video with button which we choose (we choose 'q')
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()


