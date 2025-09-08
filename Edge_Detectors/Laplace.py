import cv2
from matplotlib import pyplot as plt
img = cv2.imread('img.jpeg', cv2.IMREAD_GRAYSCALE)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')
plt.show()
