import cv2 as cv


img = cv.imread('Photos/pamukkale.jpg')
cv.imshow('Pamukkale',img)

''' Convert in cv library'''
# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# BGR to HSV
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow('HSV',hsv)

# BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB',lab)

# BGR to RGB
rgb =cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB',rgb)

# HSV to BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV ==> BGR',hsv_bgr)

cv.waitKey(0)