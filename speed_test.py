import numpy as np
from numba import njit
import ray

ray.init(num_cpus=32)

# njit()

def test(runs=1):
    value = np.zeros(100, dtype=np.uint64)
    for i in range(4):
        value += 2 * 16**i
    mask = np.uint64(0xFFFF)
    for i in range(runs):
        value + mask

def test2(runs=1, split=10):
    obj = [
        fn.remote(runs//split) if i != 0 else fn.remote(runs//split + runs % split) for i in range(split)
    ]
    data = ray.get(obj)

# njit()
@ray.remote
def fn(runs=1):
    value = np.zeros(100, dtype=np.uint64)
    for i in range(4):
        value += 2 * 16**i
    mask = np.uint64(0xFFFF)
    for i in range(runs):
        value + mask

if __name__ == '__main__':
    import timeit
    import sys
    print("testing first run")
    print(timeit.timeit('test()', globals=globals(), number=1))
    print("testing 10000000 runs")
    print(timeit.timeit('test(10000000)', globals=globals(), number=1))

    print("testing first run")
    print(timeit.timeit('test2()', globals=globals(), number=1))
    print("testing 10000000 runs")
    print(timeit.timeit('test2(10000000)', globals=globals(), number=1))