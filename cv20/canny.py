from gaussian import GaussianBlur
from kernel import KERNEL_SOBEL
from convolution import convolution
import cv2
import numpy as np


def Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None):
    gray_scaled_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # blur the image
    blurred_image = GaussianBlur(gray_scaled_image, 5, 1.4)

    # find magnitude and orientation of gradient
    kx, ky = KERNEL_SOBEL
    Ix = convolution(blurred_image, kx)
    Iy = convolution(blurred_image, ky)

    G = np.hypot(Ix, Iy)

    # Non-maximum suppression

    # Linking and thresholding
    return image
