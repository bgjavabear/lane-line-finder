from gaussian import GaussianBlur
from kernel import KERNEL_SOBEL
import cv2
import numpy as np


def Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None):
    gray_scaled_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # blur the image
    blurred_image = GaussianBlur(gray_scaled_image, 5, 1.4)

    kx, ky = KERNEL_SOBEL

    # find magnitude and orientation of gradient

    # Non-maximum suppression

    # Linking and thresholding
    return image
