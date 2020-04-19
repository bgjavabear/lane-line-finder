import numpy as np


def normalize(arr):
    return arr / np.sum(arr)


def is_odd_number(number):
    return number % 2 == 1
