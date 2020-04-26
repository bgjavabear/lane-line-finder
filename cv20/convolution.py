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


def convolution(src, k, borderType=BorderType.REPLICATE):
    out = np.zeros_like(src, dtype=float)
    s = k.shape[0] // 2
    srct = transform_image(src, s, borderType)
    xu = srct.shape[1]
    yu = srct.shape[0]

    for x in range(s, xu - s):
        for y in range(s, yu - s):
            apply_filter(x - s, y - s, k, srct[y - s:y + s + 1, x - s:x + s + 1], out)

    return out


def apply_filter(x, y, kernel, target, out):
    out[y][x] = np.sum(kernel * target)


def transform_image(src, pad_width, borderType):
    if borderType == BorderType.CLIP:
        return np.pad(src, pad_width, mode='constant', constant_values=0)
    if borderType == BorderType.REPLICATE:
        return np.pad(src, pad_width, mode='edge')
    raise Exception('Unsupported border type')
