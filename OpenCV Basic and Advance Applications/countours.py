import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.jpg')
cv.imshow('Cats',img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank',blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Cat',gray)

# METHOD 1

blur =cv.GaussianBlur(gray,(3,3),cv.BORDER_DEFAULT)
cv.imshow('Blurred Cat',blur)

canny = cv.Canny(blur, 125,175)
cv.imshow('Canny Edges',canny)

# METHOD 2

ret, thresh = cv.threshold(gray,125,255, cv.THRESH_BINARY)
cv.imshow('Thresholded Cat',thresh)


contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found !!!')

cv.drawContours(blank, contours, -1, (0,255,0), 1)
cv.imshow('Contours Blank',blank)

cv.waitKey(0)