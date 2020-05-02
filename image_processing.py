import math
import cv2
import numpy as np


def finding_lane_lines(src):
    src_shape = src.shape
    # parameters
    kernel_size = 5
    sigma = 1.4
    canny_threshold_min = 100
    canny_threshold_max = 150
    rho = 1
    theta = np.pi / 180
    threshold = 25
    min_line_length = 40
    max_line_gap = 20
    polygon_vertex1, polygon_vertex2, polygon_vertex3, polygon_vertex4 = get_vertices(src_shape, .9, .1, .39)

    # gray scale
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    gray = gray + s

    # blur
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

    # canny
    edges = cv2.Canny(blur, canny_threshold_min, canny_threshold_max)

    # region
    mask = np.zeros_like(edges)

    vertices = np.array([[polygon_vertex1, polygon_vertex2, polygon_vertex3, polygon_vertex4]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 1)
    masked_edges = cv2.bitwise_and(edges, mask)

    # hough
    line_image = np.zeros_like(src)
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    right_x1_arr, right_y1_arr = np.array([]), np.array([])
    right_x2_arr, right_y2_arr = np.array([]), np.array([])

    left_x1_arr, left_y1_arr = np.array([]), np.array([])
    left_x2_arr, left_y2_arr = np.array([]), np.array([])

    for line in lines:
        for x1, y1, x2, y2 in line:
            rad = math.atan2(y2 - y1, x2 - x1)
            angle = rad * 180 / math.pi
            if -40 < angle < -20:
                x1_adjusted, y1_adjusted = int(x1 - (src_shape[0] - y1) / math.tan(-rad)), src_shape[0]
                x2_adjusted, y2_adjusted = int(x1 + y1 / math.tan(-rad)), 0
                right_x1_arr = np.append(right_x1_arr, x1_adjusted)
                right_y1_arr = np.append(right_y1_arr, y1_adjusted)
                right_x2_arr = np.append(right_x2_arr, x2_adjusted)
                right_y2_arr = np.append(right_y2_arr, y2_adjusted)
            elif 20 < angle < 40:
                x1_adjusted, y1_adjusted = int(x2 - y2 / math.tan(rad)), 0
                x2_adjusted, y2_adjusted = int(x2 + (src_shape[0] - y2) / math.tan(rad)), src_shape[0]
                left_x1_arr = np.append(left_x1_arr, x1_adjusted)
                left_y1_arr = np.append(left_y1_arr, y1_adjusted)
                left_x2_arr = np.append(left_x2_arr, x2_adjusted)
                left_y2_arr = np.append(left_y2_arr, y2_adjusted)

    if right_x1_arr.size != 0 and right_y1_arr.size != 0 and right_x2_arr.size != 0 and right_y2_arr.size != 0:
        cv2.line(line_image, (int(np.average(right_x1_arr)), int(np.average(right_y1_arr))),
                 (int(np.average(right_x2_arr)), int(np.average(right_y2_arr))), (255, 0, 0), 10)

    if left_x1_arr.size != 0 and left_y1_arr.size != 0 and left_x2_arr.size != 0 and left_y2_arr.size != 0:
        cv2.line(line_image, (int(np.average(left_x1_arr)), int(np.average(left_y1_arr))),
                 (int(np.average(left_x2_arr)), int(np.average(left_y2_arr))), (255, 0, 0), 10)

    line_image[:, :, 0] = cv2.bitwise_and(line_image[:, :, 0], mask) * 255

    line_edges = cv2.addWeighted(src, 0.8, line_image, 1, 0)
    return line_edges


def get_vertices(shape, lower_width_percentage, upper_width_percentage, height_percentage):
    y_max = shape[0]
    x_max = shape[1]

    height = int(y_max * (1 - height_percentage))
    lower_width = int(x_max * lower_width_percentage)
    upper_width = int(x_max * upper_width_percentage)
    x_lower_delta = (x_max - lower_width) // 2
    x_upper_delta = (lower_width - upper_width) // 2 + x_lower_delta

    polygon_vertex1 = (x_lower_delta, y_max)
    polygon_vertex2 = (x_upper_delta, height)
    polygon_vertex3 = (x_max - x_upper_delta, height)
    polygon_vertex4 = (x_max - x_lower_delta, y_max)
    return [polygon_vertex1, polygon_vertex2, polygon_vertex3, polygon_vertex4]
