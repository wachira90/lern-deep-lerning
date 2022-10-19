# test gpu

## test code numba

````
#! python
from numba import jit
import numpy as np
from timeit import default_timer as timer

# To run on CPU
def func(a):
    for i in range(10000000):
        a[i]+= 1

# To run on GPU
@jit
def func2(x):
    return x+1

if __name__=="__main__":
    n = 10000000
    a = np.ones(n, dtype = np.float64)
    start = timer()
    func(a)
    print("without GPU:", timer()-start)
    start = timer()
    func2(a)
    numba.cuda.profile_stop()
    print("with GPU:", timer()-start)
````    

## example 2

````   
#! python
from numba import cuda
import numpy as np
import time

@cuda.jit
def hello(data):
    data[cuda.blockIdx.x, cuda.threadIdx.x] = cuda.blockIdx.x

numBlocks = 5
threadsPerBlock = 10

data = np.ones((numBlocks, threadsPerBlock), dtype=np.uint8)

hello[numBlocks, threadsPerBlock](data)

print(data)
````   
