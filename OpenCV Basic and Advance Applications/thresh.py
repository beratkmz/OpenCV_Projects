import cv2 as cv

img = cv.imread('Photos/pamukkale.jpg')
cv.imshow('Pamukkale',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# Simple Thresholding
threshold, thresh = cv.threshold(gray,150,255,cv.THRESH_BINARY)
cv.imshow('Simple Thresholded',thresh)

threshold, thresh_inv = cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
cv.imshow('Simple Thresholded Inverse',thresh_inv)

# Adaptive Thresholding
adaptive_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,19,3)
cv.imshow('Adaptive Thresholding',adaptive_thresh)

adaptive_thresh_inv = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,19,3)
cv.imshow('Adaptive Thresholding Inverse',adaptive_thresh_inv)

cv.waitKey(0)