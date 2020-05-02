from gaussian import GaussianBlur
from kernel import KERNEL_SOBEL
from convolution import convolution
import cv2
import numpy as np


def Canny(image, threshold1=0.05, threshold2=0.09):
    gray_scaled_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # blur the image
    blurred_image = GaussianBlur(gray_scaled_image, 5, 1.4)

    # find magnitude and orientation of gradient
    G, theta = get_gradient_magnitude_and_orientation(blurred_image)

    # Non-maximum suppression
    non_max_suppressed_image = non_max_suppression(G, theta)

    # Double threshold
    thresholded_image, weak, strong = threshold(non_max_suppressed_image, threshold1, threshold2)

    # Edge Tracking by Hysteresis
    out = hysteresis(thresholded_image, weak, strong)

    return out.astype('uint8')


def get_gradient_magnitude_and_orientation(src):
    kx, ky = KERNEL_SOBEL
    Ix = convolution(src, kx)
    Iy = convolution(src, ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255.

    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(src, theta):
    out = np.zeros_like(src, dtype=np.int32)
    y_max, x_max = src.shape
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180  # we do not care if it is -45 or 135 degrees. Neighbours will be the same
    # arctg is between (-pi/2; pi/2), so after adding 180 degrees, we get (0, 180)

    for x in range(1, x_max - 1):
        for y in range(1, y_max - 1):
            neighbour1 = 255
            neighbour2 = 255
            if (0 <= angle[y][x] < 22.5) or (157.5 <= angle[y][x] <= 180):
                neighbour1 = src[y][x + 1]
                neighbour2 = src[y][x - 1]
            elif 22.5 <= angle[y][x] < 67.5:
                neighbour1 = src[y - 1][x + 1]
                neighbour2 = src[y + 1][x - 1]
            elif 67.5 <= angle[y][x] < 112.5:
                neighbour1 = src[y - 1][x]
                neighbour2 = src[y + 1][x]
            elif 112.5 <= angle[y][x] < 157.5:
                neighbour1 = src[y - 1][x - 1]
                neighbour2 = src[y - 1][x + 1]

            if (src[y][x] >= neighbour1) and (src[y][x] >= neighbour2):
                out[y][x] = src[y][x]
            else:
                out[y][x] = 0

    return out


def threshold(src, threshold1, threshold2):
    upper = src.max() * threshold2
    lower = upper * threshold1

    out = np.zeros_like(src)
    weak = 25
    strong = 255

    strong_y, strong_x = np.where(src >= upper)
    weak_y, weak_x = np.where((src <= upper) & (src >= lower))

    out[strong_y, strong_x] = strong
    out[weak_y, weak_x] = weak
    return out, weak, strong


def hysteresis(src, weak, strong=255):
    y_max, x_max = src.shape
    for y in range(1, y_max - 1):
        for x in range(1, x_max - 1):
            if src[y, x] == weak:
                if ((src[y + 1, x - 1] == strong) or (src[y + 1, x] == strong) or (src[y + 1, x + 1] == strong)
                        or (src[y, x - 1] == strong) or (src[y, x + 1] == strong)
                        or (src[y - 1, x - 1] == strong) or (src[y - 1, x] == strong) or (
                                src[y - 1, x + 1] == strong)):
                    src[y, x] = strong
                else:
                    src[y, x] = 0
    return src
