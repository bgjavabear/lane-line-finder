from enum import Enum
import numpy as np


class BorderType(Enum):
    """
    Enum for border types.
    CLIP: extend an image with black pixels
    COPY_EDGE(REPLICATE): extend out for each border pixel
    """
    CLIP = 1,
    REPLICATE = 2


def convolution(src, kernel, borderType=BorderType.REPLICATE):
    out = np.zeros_like(src, dtype=float)
    for x in range(0, src.shape[1]):
        for y in range(0, src.shape[0]):
            out[y][x] = get_weight_after_kernel(x, y, src, kernel, borderType)
    return out


def get_weight_after_kernel(x, y, src, kernel, borderType):
    ksize_half = int(kernel.shape[0] / 2)
    result = 0
    for kx in range(-ksize_half, ksize_half + 1):
        for ky in range(-ksize_half, ksize_half + 1):
            x_shifted = x + kx
            y_shifted = y + ky
            if is_valid(x_shifted, y_shifted, src.shape):
                result += src[y_shifted][x_shifted] * kernel[ky + ksize_half][kx + ksize_half]
            else:
                result += apply_boundary_method(x_shifted, y_shifted, src, borderType) * kernel[ky + ksize_half][
                    kx + ksize_half]
    return result


def is_valid(x, y, shape):
    return (0 <= x < shape[1]) and (0 <= y < shape[0])


def apply_boundary_method(x, y, src, borderType):
    if borderType == BorderType.CLIP:
        return 0
    if borderType == BorderType.REPLICATE:
        # upper part
        if x < 0 and y < 0:  # left upper corner
            return src[0][0]
        if y < 0 and (0 <= x < src.shape[1]):  # upper boundary
            return src[0][x]
        if y < 0 and x >= src.shape[1]:  # right upper corner
            return src[0, src.shape[1] - 1]
        # right part
        if x >= src.shape[1] and (0 <= y < src.shape[0]):
            return src[y, src.shape[1] - 1]
        # left part
        if x < 0 and (0 <= y < src.shape[0]):
            return src[y, 0]
        # lower part
        if x < 0 and y >= src.shape[0]:  # left lower corner
            return src[src.shape[0] - 1, 0]
        if y >= src.shape[0] and (0 <= x < src.shape[1]):  # lower boundary
            return src[src.shape[0] - 1, x]
        if y >= src.shape[0] and x >= src.shape[1]:  # lower right boundary
            return src[src.shape[0] - 1, src.shape[1] - 1]
        raise Exception('temporary')
    raise Exception('Unsupported border type')
