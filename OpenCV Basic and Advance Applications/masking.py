import cv2 as cv
import numpy as np

img =cv.imread('Photos/qashqai.jpg')
cv.imshow('Qashqai',img)

blank = np.zeros(img.shape[:2], dtype = 'uint8')
cv.imshow('Blank Image',blank)

mask_circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2),50,255,-1)
cv.imshow('Mask',mask_circle)

mask_rectangle = cv.rectangle(blank.copy(), (img.shape[1]//2,img.shape[0]//2),(img.shape[1]//2+ 45,img.shape[0]//2 + 45),255,-1)
cv.imshow('Mask2',mask_rectangle)

masked = cv.bitwise_and(img,img,mask= mask_circle)
cv.imshow('Masked Image',masked)

masked_2= cv.bitwise_and(img,img,mask= mask_rectangle)
cv.imshow('Masked Image 2',masked_2)


cv.waitKey(0)