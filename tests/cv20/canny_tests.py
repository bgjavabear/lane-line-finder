import numpy as np
from utils import timeit
from cv20 import get_gradient_magnitude_and_orientation


def run_all_tests():
    test_01_get_gradient_orientation()
    print('test 01 passed')


@timeit
def test_01_get_gradient_orientation():
    test_image = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 50, 50, 50],
                           [0, 0, 0, 0, 50, 150, 150],
                           [0, 0, 0, 0, 50, 150, 250]], dtype=np.float64)
    G, theta = get_gradient_magnitude_and_orientation(test_image)
    angle = theta * 180 / np.pi
    angle = angle.astype(int)

    assert angle[4][4] == -45
    assert angle[5][5] == -45
    assert angle[5][3] == 0
    assert angle[3][5] == -90


run_all_tests()
