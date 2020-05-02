import time
import numpy as np


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
