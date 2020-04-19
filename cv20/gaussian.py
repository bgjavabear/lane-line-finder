from convolution import convolution, BorderType
from math import exp
import numpy as np
from utils import normalize, is_odd_number


def GaussianBlur(src, ksize, sigmaX, sigmaY=None, borderType=BorderType.REPLICATE):
    """
    The implementation of Gaussian Blur
    """

    kernel = create_gaussian_kernel(ksize, sigmaX, sigmaY)
    return convolution(src, kernel, borderType)


def create_gaussian_kernel(ksize, sigmaX=1, sigmaY=None):
    if not is_odd_number(ksize):
        raise Exception('Kernel size = ', ksize, ". Kernel size should be an odd number.")
    if sigmaX == 0 or sigmaY == 0:
        raise Exception('Sigma cannot be equal to 0.')
    if sigmaY is None:
        sigmaY = sigmaX

    kernel = np.empty(shape=(ksize, ksize))

    ksize_half = int(ksize / 2)

    for x in range(-ksize_half, ksize_half + 1):
        for y in range(-ksize_half, ksize_half + 1):
            kernel[x + ksize_half][y + ksize_half] = exp(-1 * (x ** 2 / (2 * sigmaX ** 2) + y ** 2 / (2 * sigmaY ** 2)))

    return normalize(kernel)
