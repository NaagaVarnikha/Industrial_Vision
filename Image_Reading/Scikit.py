from skimage import io  # type: ignore
import matplotlib.pyplot as plt
img = io.imread('img2.png')            
plt.imshow(img)
plt.axis('off')
plt.title('scikit-image Image')
plt.show()