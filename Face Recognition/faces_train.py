import os
import cv2 as cv
import numpy as np

# Create image files for people which we want to recognize
people = ['Image File 1','Image File 2','Image File 3']
DIR = r'C:\Users\BERAT\Desktop\OpenCV\Faces'                        

# Calling haar cascade file
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

# Joining in every file and image documents and classfication of them, after that graying (3 channel to 1 channel) and plot rectangel and write text for detected and recognized faces
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done -------------------------')

# Creating train module files for application
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)
