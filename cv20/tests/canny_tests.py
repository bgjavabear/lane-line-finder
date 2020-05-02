import unittest
import numpy as np
from cv20 import get_gradient_magnitude_and_orientation


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_01_get_gradient_orientation(self):
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

        self.assertEqual(angle[4][4], -45)
        self.assertEqual(angle[5][5], -45)
        self.assertEqual(angle[5][3], 0)
        self.assertEqual(angle[3][5], -90)


if __name__ == '__main__':
    unittest.main()
