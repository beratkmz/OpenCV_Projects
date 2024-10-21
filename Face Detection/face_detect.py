import cv2 as cv 

''' FACE DETECTION IMAGE '''

# img = cv.imread('Photos/sena10.jpg')
# cv.imshow('Fenerbahce',img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# haar_cascade = cv.CascadeClassifier('haar_face.xml')

# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=7)

# print(f'Number of faces found = {len(faces_rect)}')

# for (x,y,w,h) in faces_rect:
#     cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

# cv.imshow('Detected Faces',img)

# cv.waitKey(0)


# ''' FACE DETECTION VIDEO '''

face_cascade = cv.CascadeClassifier('haar_face.xml')

# Webcam den video çekmeD

capture = cv.VideoCapture(0)

while True:
    # Frame okuma
    _,img = capture.read()

    # Gri skalaya çevirme ( Çoğu işlemde bu gerekli)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Yüz algılama
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=6)

    # Eğer yüz algılandıysa, yüzü dikdörtgen içine alma ve metin yazma
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.putText(img, "YUZ ALGILANDI", (20, 450), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        # Yüz algılanmadıysa metin yazma
        cv.putText(img, "YUZ ALGILANAMADI", (20, 450), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)



    # Gösterme
    cv.imshow('img',img)

    # Atanan tuşla videoyu durdurma ('d' tuşuna atandı)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()

