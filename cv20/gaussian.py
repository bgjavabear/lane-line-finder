from convolution import convolution, BorderType
from kernel import create_gaussian_kernel


def GaussianBlur(src, ksize, sigmaX, sigmaY=None, borderType=BorderType.REPLICATE):
    """
    The implementation of Gaussian Blur
    """

    kernel = create_gaussian_kernel(ksize, sigmaX, sigmaY)
    return convolution(src, kernel, borderType)
