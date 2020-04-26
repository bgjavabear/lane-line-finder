import numpy as np

image = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]])

x_center, y_center = 2, 2
x_upper, y_upper = image.shape[1], image.shape[0]
width = 1

image_x = image[x_center - width:x_center + width + 1, y_center - width:y_center + width + 1]

kernel = np.array([[2, 2, 2],
                   [2, 2, 2],
                   [2, 2, 2]])

result = np.sum(kernel * image_x)

print("asdasd")
