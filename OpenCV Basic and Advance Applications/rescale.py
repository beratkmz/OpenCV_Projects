import cv2 as cv

""" Reading Photo """

img = cv.imread('Photos/cat.jpg')
cv.imshow('Cat',img)

# Rescaling Settings

def rescaleFrame(frame,scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0]* scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

# Showing Resized Photo

resized_image = rescaleFrame(img)
cv.imshow('Resized Image', resized_image )

# Just for Live Videos
def changeRes(width,height):
    capture.set(3,width)
    capture.set(4,height)

""" Reading Videos """

capture = cv.VideoCapture('Videos/123.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame)                        # Describe resized frame

    cv.imshow('Video', frame)
    cv.imshow('Resized Video', frame_resized )                 # Showing Resized Video           

    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
