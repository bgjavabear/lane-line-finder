import numpy as np
from cv20 import convolution, BorderType
from utils import timeit


def run_all_tests():
    test_01_basic_kernel_3x3()
    print('test 01 passed')
    test_02_basic_kernel_5x5()
    print('test 02 passed')
    test_03_custom_kernel_3x3()
    print('test 03 passed')


@timeit
def test_01_basic_kernel_3x3():
    image = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0]])

    kernel = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
    output = convolution(image, kernel, BorderType.CLIP)
    assert np.array_equal(image, output)


@timeit
def test_02_basic_kernel_5x5():
    image = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0]])

    kernel = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
    output = convolution(image, kernel, BorderType.CLIP)
    assert np.array_equal(image, output)


@timeit
def test_03_custom_kernel_3x3():
    image = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0]])

    kernel = np.full((3, 3), 1 / 9)

    output_expected = np.array([[0, 0, 1 / 9, 1 / 9, 1 / 9],
                                [0, 1 / 9, 2 / 9, 2 / 9, 1 / 9],
                                [0, 2 / 9, 3 / 9, 3 / 9, 1 / 9],
                                [0, 2 / 9, 2 / 9, 2 / 9, 0],
                                [0, 1 / 9, 1 / 9, 1 / 9, 0]])

    output_actual = convolution(image, kernel, BorderType.CLIP)
    assert np.array_equal(output_expected, output_actual)


run_all_tests()
