import cv2 as cv
import time

# Introduction of modules to be used (Face and Mouth)
face_cascade = cv.CascadeClassifier('haar_face.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# Capturing video from webcam
capture = cv.VideoCapture(0)

# Definitions for calculating FPS and running at a certain time threshold
prev_time = time.time()
fps_display_time = time.time()
fps_display_interval = 0.5
fps_to_display = 0

# Blink counter and threshold value
blink_counter = 0
eye_closed = False
EYE_DETECT_TIMEOUT = 0.4  # Göz algılanmama süresi (saniye)
last_eye_detection_time = time.time()  # Göz algılanma zamanı

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

    # Eye detection
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            roi_gray = gray[y:y + h, x:x + w]  
            roi_color = img[y:y + h, x:x + w]  

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=40)

            if len(eyes) >= 2:  # If 2 eyes detected
                last_eye_detection_time = time.time()  
                eye_closed = False
                for (ex, ey, ew, eh) in eyes:
                    cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                draw_centered_text_bottom(img, "EYES DETECTED", cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:  
                if (time.time() - last_eye_detection_time) >= EYE_DETECT_TIMEOUT:
                    if not eye_closed: 
                        blink_counter += 1
                        eye_closed = True
                    draw_centered_text_bottom(img, "BLINK DETECTED", cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    else:
        draw_centered_text_bottom(img, "FACE NOT DETECTED", cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Updating FPS at a certain time threshold
    if curr_time - fps_display_time >= fps_display_interval:
        fps_to_display = int(fps)
        fps_display_time = curr_time

    # Showing FPS and blink counter
    cv.putText(img, f"FPS: {fps_to_display}", (img.shape[1] - 150, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.putText(img, f"Blinks: {blink_counter}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv.imshow('Live Camera', img)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
