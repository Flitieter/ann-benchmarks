from ..base.module import BaseANN

import ctypes
import numpy

c_module = ctypes.CDLL('ann_benchmarks/algorithms/alaya/libAlayaDB.so')

c_fit = c_module.fit
c_fit.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
c_fit.restype = ctypes.c_void_p

c_batch_query = c_module.batch_query
c_batch_query.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
c_batch_query.restype = ctypes.c_void_p


class Alaya(BaseANN):
    def __init__(self, metric, test_param=0):
        self._test_param = test_param
        self._metric = metric
        self.name = "Alaya(test_param=%d)" % self._test_param
        # self.data = None
        self.ef = 32
        self.rerank = 20

    def fit(self, X: numpy.array):
        print('fit starts')
        rows, cols = X.shape
        X_flat = X.flatten()
        X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # self.data  = X_ctypes

        c_fit(X_ctypes, rows, cols)

        print('fit done')
    
    def query(self, v: numpy.array, n: int):
        raise NotImplementedError

    def set_query_arguments(self, ef):
        self.ef, self.rerank = ef

    def batch_query(self, X: numpy.array, n: int) -> None:
        # print('batch_query starts')
        # print(X)
        rows, cols = X.shape
        X_flat = X.flatten()
        X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        A_ctypes = (ctypes.c_uint32 * (rows * n))()

        # print('python c_batch')
        c_batch_query(X_ctypes, rows, cols, n, self.ef, self.rerank, A_ctypes)

        self.res = numpy.frombuffer(A_ctypes, dtype=numpy.uint32).reshape(rows, n)

        # print('batch_query done')
        # print(self.res)

def main():
    dataset = numpy.array([[1,2,3,4,5], [1,2,2,3,5]], dtype=numpy.float32)
    algo = Alaya(0)
    algo.fit(dataset)
    queries = numpy.array([[1,2,2,3,5]], dtype=numpy.float32)
    algo.batch_query(queries, 1)

    # ans = numpy.array([[i, i + 1] for i in range(5)], dtype=numpy.uint32)
    # assert ans == algo.res
    

if __name__ == '__main__':
    main()

        


