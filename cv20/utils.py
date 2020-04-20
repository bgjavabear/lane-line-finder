import numpy as np
import time
import math


def normalize(arr):
    return arr / np.sum(arr)


def is_odd_number(number):
    return number % 2 == 1


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()

        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def squared_diff(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise Exception('Shapes are different')

    result = 0.
    for x in range(0, arr1.shape[1]):
        for y in range(0, arr1.shape[0]):
            result += math.sqrt((arr2[y][x] - arr1[y][x]) ** 2)
    return result
