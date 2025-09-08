import cv2
img = cv2.imread('img1.png')          
cv2.imshow('OpenCV Image', img)         
cv2.waitKey(0)
cv2.destroyAllWindows()