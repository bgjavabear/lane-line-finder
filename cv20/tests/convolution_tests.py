import unittest
import numpy as np
from cv20 import BorderType, convolution, transform_image


class MyTestCase(unittest.TestCase):
    def test_01_basic_kernel_3x3(self):
        image = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0]])

        kernel = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
        output = convolution(image, kernel, BorderType.CLIP)
        self.assertTrue(np.array_equal(image, output))

    def test_02_basic_kernel_5x5(self):
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
        self.assertTrue(np.array_equal(image, output))

    def test_03_custom_kernel_3x3(self):
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

        self.assertTrue(np.array_equal(output_actual, output_expected))

    def test_04_transform_image_clip(self):
        test = np.array([[1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5]])

        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 2, 3, 4, 5, 0, 0],
                             [0, 0, 1, 2, 3, 4, 5, 0, 0],
                             [0, 0, 1, 2, 3, 4, 5, 0, 0],
                             [0, 0, 1, 2, 3, 4, 5, 0, 0],
                             [0, 0, 1, 2, 3, 4, 5, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        actual = transform_image(test, 2, BorderType.CLIP)
        self.assertTrue(np.array_equal(expected, actual))

    def test_05_transform_image_replicate(self):
        test = np.array([[1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5]])

        expected = np.array([[1, 1, 1, 2, 3, 4, 5, 5, 5],
                             [1, 1, 1, 2, 3, 4, 5, 5, 5],
                             [1, 1, 1, 2, 3, 4, 5, 5, 5],
                             [1, 1, 1, 2, 3, 4, 5, 5, 5],
                             [1, 1, 1, 2, 3, 4, 5, 5, 5],
                             [1, 1, 1, 2, 3, 4, 5, 5, 5],
                             [1, 1, 1, 2, 3, 4, 5, 5, 5],
                             [1, 1, 1, 2, 3, 4, 5, 5, 5],
                             [1, 1, 1, 2, 3, 4, 5, 5, 5]])
        actual = transform_image(test, 2, BorderType.REPLICATE)
        self.assertTrue(np.array_equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
