import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('img1.png')         
plt.imshow(img)
plt.axis('off')
plt.title('matplotlib Image')
plt.show()