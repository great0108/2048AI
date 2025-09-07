import numpy as np
from numba import njit


# njit()
def test(runs=1):
    value = np.zeros(100, dtype=np.uint64)
    for i in range(4):
        value += 2 * 16**i
    mask = np.uint64(0xFFFF)
    for i in range(runs):
        value + mask

# njit()
def test2(runs=1):
    value = np.zeros((100, 4), dtype=np.uint16)
    for i in range(4):
        for j in range(4):
            value[:, i] += 2 * 16**j

    for i in range(runs):
        value[:, 0]

if __name__ == '__main__':
    import timeit
    import sys
    print("testing first run")
    print(timeit.timeit('test()', globals=globals(), number=1))
    print("testing 10000000 runs")
    print(timeit.timeit('test(10000000)', globals=globals(), number=1))

    # print("testing first run")
    # print(timeit.timeit('test2()', globals=globals(), number=1))
    # print("testing 10000000 runs")
    # print(timeit.timeit('test2(10000000)', globals=globals(), number=1))