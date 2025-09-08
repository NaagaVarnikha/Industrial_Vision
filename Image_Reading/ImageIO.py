import imageio.v2 as imageio # type: ignore
import matplotlib.pyplot as plt
img = imageio.imread('img2.png')       
plt.imshow(img)                         
plt.axis('off')
plt.title('imageio Image')
plt.show()