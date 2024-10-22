import cv2 as cv
import time

# Introduction of modules to be used (Face and Mouth)
face_cascade = cv.CascadeClassifier('haar_face.xml')
mouth_cascade = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')

# Capturing video from webcam
capture = cv.VideoCapture(0)

# Definitions for calculating FPS and running at a certain time threshold
prev_time = time.time()         
fps_display_time = time.time()  
fps_display_interval = 0.5      
fps_to_display = 0              

# Definition for the text which centered at the bottom
def draw_centered_text_bottom(img, text, font, scale, color, thickness):
    text_size = cv.getTextSize(text, font, scale, thickness)[0]
    img_height, img_width = img.shape[:2]
    text_x = (img_width - text_size[0]) // 2
    text_y = img_height - 50
    cv.putText(img, text, (text_x, text_y), font, scale, color, thickness)

while True:
    # Frame reading
    _, img = capture.read()

    # BGR 3 channel to 1 Gray channel
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    # Mouth detection
    mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=100)

    # Drawing and texting for (un)detected faces
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        draw_centered_text_bottom(img, "FACE DETECTED", cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        draw_centered_text_bottom(img, "FACE NOT DETECTED", cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)


    # Drawing for detected mouths and giving alert for users against to usage of masks for Covid-19
    if len(mouth) > 0:
        for (x, y, w, h) in mouth:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            cv.putText(img, "!! TAKE YOUR MASK !!", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Updating FPS at a certain time threshold
    if curr_time - fps_display_time >= fps_display_interval:
        fps_to_display = int(fps)
        fps_display_time = curr_time

    # Showing FPS
    cv.putText(img, f"FPS: {fps_to_display}", (img.shape[1] - 150, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    cv.imshow('Live Camera', img)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
