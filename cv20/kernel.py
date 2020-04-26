import numpy as np

from utils import is_odd_number, normalize


def create_gaussian_kernel(ksize, sigmaX=1, sigmaY=None):
    if not is_odd_number(ksize):
        raise Exception('Kernel size = ', ksize, ". Kernel size should be an odd number.")
    if sigmaX == 0 or sigmaY == 0:
        raise Exception('Sigma cannot be equal to 0.')
    if sigmaY is None:
        sigmaY = sigmaX

    size = int(ksize) // 2

    y, x = np.mgrid[-size:size + 1, -size: size + 1]

    gk = np.exp(-(x ** 2 / (2 * sigmaX ** 2) + y ** 2 / (2 * sigmaY ** 2)))
    return normalize(gk)


def get_sobel_kernels():
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    ky = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    return kx, ky


KERNEL_SOBEL = get_sobel_kernels()
