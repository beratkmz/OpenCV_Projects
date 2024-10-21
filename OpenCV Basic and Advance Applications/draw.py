import cv2 as cv
import numpy as np

# Creating blank image

blank = np.zeros((500,500,3), dtype = 'uint8')
cv.imshow('Blank', blank)


# # 1. Paint the image a certain colour
# blank[200:300 , 300:400] = 0,0,255
# cv.imshow('Red',blank)

# # 2. Draw a Rectangle
# cv.rectangle (blank, (0,0), (250,250), (0,255,0), -1)
# cv.imshow('Rectangle',blank)

# 3. Draw a circle 
cv.circle(blank,(200,100), 50, (0,0,255), -1)
cv.circle(blank,(300,100),50,(0,0,255),-1)
cv.circle(blank,(250,140),50,(0,0,255),-1)
# cv.imshow('Circle',blank)

# 4. Draw a line 
cv.line(blank,(190,140),(250,200),(0,0,255),25)
cv.line(blank,(310,140),(250,200),(0,0,255),25)
cv.line(blank,(250,100),(250,200),(0,0,255),25)
# cv.imshow('Line',blank)

# 5. Write text
cv.putText(blank,'SENI COK SEVIYORUM',(85,300),cv.FONT_HERSHEY_TRIPLEX, 1.0,(0,0,255), 2)
cv.imshow('Text',blank)

cv.waitKey(0)