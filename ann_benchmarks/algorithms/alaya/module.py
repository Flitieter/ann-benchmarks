import ctypes
import numpy as np

from ..base.module import BaseANN

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

    def fit(self, X: np.ndarray):
        print('fit starts')
        rows, cols = X.shape

        X_flat = X.flatten()

        X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # self.data  = X_ctypes

        c_fit(X_ctypes, rows, cols)

        print('fit done')
    
    def query(self, v: np.ndarray, n: int):
        rows, cols = v.shape
        X_flat = v.flatten()
        X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        A_ctypes = (ctypes.c_uint32 * (rows * n))()

        # print('python c_batch')
        c_batch_query(X_ctypes, rows, cols, n, self.ef, self.rerank, A_ctypes)

        self.res = np.frombuffer(A_ctypes, dtype=np.uint32).reshape(rows, n)
        # raise NotImplementedError

    def set_query_arguments(self, ef:int):
        self.ef, self.rerank = ef

    # def batch_query(self, X: numpy.array, n: int) -> None:
    #     # print('batch_query starts')
    #     # print(X)
    #     rows, cols = X.shape
    #     X_flat = X.flatten()
    #     X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    #     A_ctypes = (ctypes.c_uint32 * (rows * n))()

    #     # print('python c_batch')
    #     c_batch_query(X_ctypes, rows, cols, n, self.ef, self.rerank, A_ctypes)

    #     self.res = numpy.frombuffer(A_ctypes, dtype=numpy.uint32).reshape(rows, n)

    #     # print('batch_query done')
    #     # print(self.res)

def main():
    import random
    import time
    random.seed(time.time())
    dataset = np.array([np.random.rand(100) * 10 for i in range(100000)], dtype=np.float32)
    algo = Alaya(0)
    algo.fit(dataset)
    queries = np.array([np.random.rand(100) * 10 for i in range(100)], dtype=np.float32)
    algo.batch_query(queries, 1)

    # ans = numpy.array([[i, i + 1] for i in range(5)], dtype=numpy.uint32)
    # assert ans == algo.res
    

if __name__ == '__main__':
    main()

        


