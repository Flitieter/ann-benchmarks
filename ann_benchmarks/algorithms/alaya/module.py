import numpy as np
import ctypes
from ctypes import c_void_p, c_long, c_int, POINTER
import os
from ..base.module import BaseANN

c_module = ctypes.CDLL('ann_benchmarks/algorithms/alaya/libAlayaDB.so')

c_init_alaya = c_module.init_alaya
c_init_alaya.argtypes = []
c_init_alaya.restype = ctypes.c_void_p

c_del_alaya = c_module.del_alaya
c_del_alaya.argtypes = [ctypes.c_void_p]
c_del_alaya.restype = None

c_fit = c_module.fit
c_fit.argtypes = [c_void_p, c_void_p, POINTER(c_long), POINTER(c_long), c_int]
c_fit.restype = None

c_query = c_module.query
c_query.argtypes = [c_void_p, c_void_p, POINTER(c_long), POINTER(c_long), c_int, c_int, c_int, c_void_p]
c_query.restype = None


class Alaya(BaseANN):
    def __init__(self, metric, M):
        self._metric = metric
        self._M = M
        self.name = "Alaya(metric={}, M={})".format(metric, M)
        # self.data = None
        self.ef = 32
        self.rerank = 20
        self.dim = None
        self._data = None
        self.c_alaya = c_init_alaya()

    def fit(self, X: np.ndarray):
        X = X.astype(np.float32)
        self.X = X
        c_fit(self.c_alaya, X.ctypes.data, X.ctypes.shape, X.ctypes.strides, self._M)
        # print('fit starts')
        # rows, cols = X.shape
        # self.dim = cols

        # X_flat = X.flatten()

        # print('flatten done')
        # X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # self._data = X_ctypes

        # c_fit(X_ctypes, rows, cols, self._M)
        # print('fit done')


    def query(self, v: np.ndarray, n: int):
        v = v.reshape(1, -1).astype(np.float32)
        res = np.zeros(n, dtype=np.int64)
        c_query(self.c_alaya, v.ctypes.data, v.ctypes.shape, v.ctypes.strides, n, self.ef, self.rerank, res.ctypes.data)
        return res
        # v_ctypes = v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # q_res = (ctypes.c_uint32 * n)()
        # c_query(v_ctypes, self._data, self.dim, n, self.ef, self.rerank, q_res)

        # return np.frombuffer(q_res, dtype=np.uint32)

    def set_query_arguments(self, ef):
        self.ef, self.rerank = ef

    # def batch_query(self, X: np.array, n: int) -> None:
    #     rows, cols = X.shape
    #     X_flat = X.flatten()
    #     X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    #     A_ctypes = (ctypes.c_uint32 * (rows * n))()

    #     c_batch_query(X_ctypes, self._data, rows, cols, n, self.ef, self.rerank, A_ctypes)

        # self.res = np.frombuffer(A_ctypes, dtype=np.uint32).reshape(rows, n)

# def main():
#     import random
#     import time
#     random.seed(time.time())
#     dataset = np.array([np.random.rand(100) * 10 for i in range(100000)], dtype=np.float32)
#     algo = Alaya(0)
#     algo.fit(dataset)
#     queries = np.array([np.random.rand(100) * 10 for i in range(100)], dtype=np.float32)
#     algo.batch_query(queries, 1)

#     # ans = numpy.array([[i, i + 1] for i in range(5)], dtype=numpy.uint32)
#     # assert ans == algo.res
    

# if __name__ == '__main__':
#     main()

# class Alaya(BaseANN):
#     def __init__(self, metric, method_param) -> None:
#         self._metric = metric
#         self._efConstruction = method_param['efConstruction']
#         self._M = method_param['M']
#         self._level = method_param['level']
#         self._name = 'Alaya_(%s)' % (method_param)
#         self._dir = 'alaya_indices'
#         self._index_path = f'M_{self._M}_efCon{self._efConstruction}_level{self._level}.hnsw'

#     def fit(self, X: np.array) -> None:
#         self.dim = X.shape[1]
#         if not os.path.exists(self._dir):
#             os.mkdir(self._dir)
#         if self._index_path not in os.listdir(self._dir):
#             index = alaya.Index("HNSW", dim=self.dim, metric=self._metric, M=self._M, L=self._efConstruction)
#             self._graph = index.build(X)
#             self._graph.save(os.path.join(self._dir, self._index_path))
#         self._graph = alaya.Graph(os.path.join(self._dir, self._index_path))
#         self._searcher = alaya.Searcher(self._graph, X, self._metric, self._level)
#         self._searcher.optimize(1)


#     def set_query_arguments(self, ef):
#         self._searcher.set_ef(ef)

#     def prepare_query(self, q: np.array, n: int) -> None:
#         self.q = q
#         self.n = n

#     def run_prepared_query(self):
#         self.res = self._searcher.search(self.q, self.n)

#     def get_prepared_query_results(self):
#         return self.res


# c_fit = c_module.fit
# c_fit.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
# c_fit.restype = None

# c_query = c_module.query
# c_query.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
#                     ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
# c_query.restype = None

# c_batch_query = c_module.batch_query
# c_batch_query.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
#                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
# c_batch_query.restype = None
