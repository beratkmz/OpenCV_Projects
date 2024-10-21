import cv2 as cv
import numpy as np
import pyautogui

# Haar Cascade and LBPH Moduls
haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ['Image File 1', 'Image File 2', 'Image Fİle 3']

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

region = (1200, 100, 640, 480)
while True:
    # Taking screenshot
    screenshot = pyautogui.screenshot(region=region)

    # Convert screenshot to numpy array
    img = np.array(screenshot)

    # Convert BGR form (pyautogui is RGB, OpenCV use BGR)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(img, str(people[label]), (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow('WhatsApp Video Call Face Detection', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
