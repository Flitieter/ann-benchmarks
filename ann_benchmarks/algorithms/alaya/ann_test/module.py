from module_bace import BaseANN

import numpy
import ctypes
import random
import time


c_module = ctypes.CDLL('../build/libhybrid_search.so')

c_fit = c_module.fit
c_fit.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
c_fit.restype = ctypes.c_void_p

c_batch_query = c_module.batch_query
c_batch_query.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
c_batch_query.restype = ctypes.c_void_p


class Alaya(BaseANN):
    def __init__(self, metric, test_param=0):
        self._test_param = test_param
        self._metric = metric
        self.name = "Alaya(test_param=%d)" % self._test_param

    def fit(self, X: numpy.array):
        print('fit starts')
        print(X)
        rows, cols = X.shape
        X_flat = X.flatten()
        X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        c_fit(X_ctypes, rows, cols)

        print('fit done')
    
    def query(self, v: numpy.array, n: int):
        raise NotImplementedError

    def batch_query(self, X: numpy.array, n: int) -> None:
        print('batch_query starts')
        print(X)
        rows, cols = X.shape
        X_flat = X.flatten()
        X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        A_ctypes = (ctypes.c_uint32 * (rows * n))()

        c_batch_query(X_ctypes, rows, cols, n, A_ctypes)

        self.res = numpy.frombuffer(A_ctypes, dtype=numpy.uint32)

        print('batch_query done')
        print(self.res)


def main():
    random.seed(time.time())
    dataset = numpy.array([numpy.random.uniform(0,1000000,100) for i in range(100000)], dtype=numpy.float32)
    algo = Alaya(0)
    algo.fit(dataset)
    queries = numpy.array([numpy.random.uniform(0,1000000,100) for i in range(5)], dtype=numpy.float32)
    algo.batch_query(queries, 100)
    ans = numpy.array([[i, i + 1] for i in range(5)], dtype=numpy.uint32)
    assert ans == algo.res
    

if __name__ == '__main__':
    main()

        

        

        
