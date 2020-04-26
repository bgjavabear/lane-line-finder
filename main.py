from cv20 import convolution
from cv20 import GaussianBlur
from cv20.kernel import KERNEL_SOBEL
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

image = mpimg.imread('tests/cv20/resources/margo_test.jpg')

gray_scaled_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# blur the image
blurred_image = GaussianBlur(gray_scaled_image, 5, 1.4)

# find magnitude and orientation of gradient
kx, ky = KERNEL_SOBEL
Ix = convolution(blurred_image, kx)
Iy = convolution(blurred_image, ky)

G = np.hypot(Ix, Iy)
f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(20, 10))
ax1.set_title('Original')
ax1.imshow(image)
ax2.set_title('Gray Scaled')
ax2.imshow(gray_scaled_image, cmap='gray')
ax3.set_title('Blurred')
ax3.imshow(blurred_image, cmap='gray')
ax4.set_title('Gradient x')
ax4.imshow(Ix, cmap='gray')
ax5.set_title('Gradient y')
ax5.imshow(Iy, cmap='gray')
ax6.set_title('Gradient')
ax6.imshow(G, cmap='gray')

plt.show()
