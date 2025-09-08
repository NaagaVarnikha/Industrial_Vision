import cv2
from matplotlib import pyplot as plt
img = cv2.imread('img.jpeg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100, 200)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.show()
