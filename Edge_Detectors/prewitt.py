import cv2
from matplotlib import pyplot as plt
import numpy as np
img = cv2.imread('img.jpeg', cv2.IMREAD_GRAYSCALE)
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewittx = cv2.filter2D(img, -1, kernelx)
prewitty = cv2.filter2D(img, -1, kernely)
prewitt_combined = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)
plt.subplot(1,3,1), plt.imshow(prewittx, cmap='gray'), plt.title('Prewitt X')
plt.subplot(1,3,2), plt.imshow(prewitty, cmap='gray'), plt.title('Prewitt Y')
plt.subplot(1,3,3), plt.imshow(prewitt_combined, cmap='gray'), plt.title('Combined')
plt.show()