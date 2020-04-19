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
    out = np.empty_like(src)
    for x in range(0, src.shape[0]):
        for y in range(0, src.shape[1]):
            out[x][y] = get_weight_after_kernel(x, y, src, kernel, borderType)
    return out


def get_weight_after_kernel(x, y, src, kernel, borderType):
    ksize_half = int(kernel.shape[0] / 2)
    result = 0
    for kx in range(-ksize_half, ksize_half + 1):
        for ky in range(-ksize_half, ksize_half + 1):
            x_shifted = x + kx
            y_shifted = y + ky
            if is_valid(x_shifted, y_shifted, src.shape):
                result += src[x_shifted][y_shifted] * kernel[kx + ksize_half][ky + ksize_half]
            else:
                result += apply_boundary_method(x_shifted, y_shifted, src, borderType) * kernel[kx + ksize_half][
                    ky + ksize_half]
    return result


def is_valid(x, y, shape):
    return (0 <= x < shape[0]) and (0 <= y < shape[1])


def apply_boundary_method(x, y, src, borderType):
    if borderType == BorderType.CLIP:
        return 0
    if borderType == BorderType.REPLICATE:
        # upper part
        if x < 0 and y < 0:  # left upper corner
            return src[0][0]
        if y < 0 and (0 <= x < src.shape[0]):  # upper boundary
            return src[x][0]
        if y < 0 and x >= src.shape[0]:  # right upper corner
            return src[src.shape[0] - 1, 0]
        # right part
        if x >= src.shape[0] and (0 <= y < src.shape[1]):
            return src[src.shape[0] - 1, y]
        # left part
        if x < 0 and (0 <= y < src.shape[1]):
            return src[0, y]
        # lower part
        if x < 0 and y >= src.shape[1]:  # left lower corner
            return src[0, src.shape[1] - 1]
        if y >= src.shape[1] and (0 <= x < src.shape[0]):  # lower boundary
            return src[x, src.shape[1] - 1]
        if y >= src.shape[1] and x >= src.shape[0]:  # lower right boundary
            return src[src.shape[0] - 1, src[src.shape[1] - 1]]
        raise Exception('temporary')
    raise Exception('Unsupported border type')
