from ..base.module import BaseANN

import numpy
import ctypes


c_module = ctypes.CDLL('ann_benchmarks/algorithms/nsg/nsg.so')

c_knn_init = c_module.init_KNN_parameters
c_knn_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_knn_init.restype = ctypes.c_void_p

c_nsg_init = c_module.init_NSG_parameters
c_nsg_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_nsg_init.restype = ctypes.c_void_p


c_fit = c_module.fit
c_fit.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
c_fit.restype = ctypes.c_void_p

c_query = c_module.query
c_query.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,ctypes.POINTER(ctypes.c_uint32)]
c_query.restype = ctypes.c_void_p

c_batch_query = c_module.batch_query
c_batch_query.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,ctypes.POINTER(ctypes.c_uint32)]
c_batch_query.restype = ctypes.c_void_p

class Nsg(BaseANN):
    def __init__(self, metric ,method_param):
        self.KNN_K = method_param["KNN_K"]
        self.KNN_L = method_param["KNN_L"]
        self.KNN_iter = method_param["KNN_iter"]
        self.KNN_S = method_param["KNN_S"]
        self.KNN_R = method_param["KNN_R"]
        self.NSG_L = method_param["NSG_L"]
        self.NSG_R = method_param["NSG_R"]
        self.NSG_C = method_param["NSG_C"]

        # self.SEARCH_L = method_param["SEARCH_L"]

        self._metric = metric
        
        self.name = 'Nsg(%s)' % (method_param)
        c_knn_init(self.KNN_K, self.KNN_L, self.KNN_iter, self.KNN_S, self.KNN_R)
        c_nsg_init(self.NSG_L, self.NSG_R, self.NSG_C)

    def fit(self, X: numpy.array):
        print('fit starts')
        print(X)
        rows, cols = X.shape
        X_flat = X.flatten()
        X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        c_fit(X_ctypes, rows, cols)

        print('fit done')

    def set_query_arguments(self,search_L ):
        self.SEARCH_L = search_L
   
    def query(self, v: numpy.array, n: int):
        X_flat = v.flatten()
        X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        A_ctypes = (ctypes.c_uint32 * (n))()

        c_query(X_ctypes, self.SEARCH_L, n , A_ctypes)
        return A_ctypes
        

    def batch_query(self, X: numpy.array, n: int) -> None:
        print('batch_query starts')
        print(X)
        rows, cols = X.shape
        X_flat = X.flatten()
        X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        A_ctypes = (ctypes.c_uint32 * (rows * n))()

        c_batch_query(X_ctypes,  self.SEARCH_L, n, A_ctypes , rows)

        self.res = numpy.frombuffer(A_ctypes, dtype=numpy.uint32)

        print('batch_query done')
        print(self.res)


        

        

        
