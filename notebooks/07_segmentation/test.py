import cv2 
import numpy as np

img = cv2.imread('images/hasan.jpg')

edges = cv2.Canny(img,50,100)

cv2.imshow('my image',img)
cv2.imshow('edges',edges)

cv2.waitKey(0)
cv2.destroyAllWindows()