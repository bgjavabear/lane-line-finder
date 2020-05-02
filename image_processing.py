import cv2
import numpy as np


def finding_lane_lines(src):
    # parameters
    kernel_size = 5
    sigma = 1.4
    canny_threshold_min = 200
    canny_threshold_max = 255
    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_length = 30
    max_line_gap = 10

    # gray scale
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

    # canny
    edges = cv2.Canny(blur, canny_threshold_min, canny_threshold_max)

    # region
    mask = np.zeros_like(edges)
    ignore_mask_color = 1

    src_shape = src.shape
    vertices = np.array([[(0, src_shape[0]), (450, 290), (490, 290), (src_shape[1], src_shape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 1)
    masked_edges = cv2.bitwise_and(edges, mask)

    # hough
    line_image = np.zeros_like(src)
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    line_edges = cv2.addWeighted(src, 0.8, line_image, 1, 0)
    return line_edges
