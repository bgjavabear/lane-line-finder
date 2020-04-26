from cv20 import Canny
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

image = mpimg.imread('tests/cv20/resources/margo_test.jpg')

# blur the image
out = Canny(image, 0.07, 0.15)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.set_title('Original')
ax1.imshow(image)
ax2.set_title('Canny')
ax2.imshow(out, cmap='gray')

plt.show()
