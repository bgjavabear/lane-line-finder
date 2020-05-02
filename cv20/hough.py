import numpy as np


def hough_line(img):
    thetas = np.deg2rad(np.arange(0., 180.))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width ** 2 + height ** 2)))
    rhos = np.linspace(0, diag_len, diag_len)

    # Cache sme reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx])
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos
