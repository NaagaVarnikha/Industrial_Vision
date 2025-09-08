import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('img.jpeg', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)
plt.subplot(1,3,1), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.subplot(1,3,2), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
plt.subplot(1,3,3), plt.imshow(sobel_combined, cmap='gray'), plt.title('Combined')
plt.show()
