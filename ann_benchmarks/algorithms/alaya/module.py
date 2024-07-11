import random
import time
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Optional
import psutil

import numpy

class BaseANN(object):
    """Base class/interface for Approximate Nearest Neighbors (ANN) algorithms used in benchmarking."""

    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        pass

    def get_memory_usage(self) -> Optional[float]:
        """Returns the current memory usage of this ANN algorithm instance in kilobytes.

        Returns:
            float: The current memory usage in kilobytes (for backwards compatibility), or None if
                this information is not available.
        """

        return psutil.Process().memory_info().rss / 1024

    def fit(self, X: numpy.array) -> None:
        """Fits the ANN algorithm to the provided data.

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            X (numpy.array): The data to fit the algorithm to.
        """
        pass

    def query(self, q: numpy.array, n: int) -> numpy.array:
        """Performs a query on the algorithm to find the nearest neighbors.

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            q (numpy.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.

        Returns:
            numpy.array: An array of indices representing the nearest neighbors.
        """
        return []  # array of candidate indices

    def batch_query(self, X: numpy.array, n: int) -> None:
        """Performs multiple queries at once and lets the algorithm figure out how to handle it.

        The default implementation uses a ThreadPool to parallelize query processing.

        Args:
            X (numpy.array): An array of vectors to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return for each query.
        Returns:
            None: self.get_batch_results() is responsible for retrieving batch result
        """
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self) -> numpy.array:
        """Retrieves the results of a batch query (from .batch_query()).

        Returns:
            numpy.array: An array of nearest neighbor results for each query in the batch.
        """
        return self.res

    def get_additional(self) -> Dict[str, Any]:
        """Returns additional attributes to be stored with the result.

        Returns:
            dict: A dictionary of additional attributes.
        """
        return {}

    def __str__(self) -> str:
        return self.name

import ctypes
import numpy

c_module = ctypes.CDLL('/home/ann-benchmark/yitao/ann-bench/ann-benchmarks/ann_benchmarks/algorithms/alaya/pyglass/cmake-build-debug/libAlayaDB.so')

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

    def set_query_arguments(self, ef:int):
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
    random.seed(time.time())
    dataset = numpy.array([numpy.random.rand(100) * 10 for i in range(100000)], dtype=numpy.float32)
    algo = Alaya(0)
    algo.fit(dataset)
    queries = numpy.array([numpy.random.rand(100) * 10 for i in range(100)], dtype=numpy.float32)
    algo.batch_query(queries, 1)

    # ans = numpy.array([[i, i + 1] for i in range(5)], dtype=numpy.uint32)
    # assert ans == algo.res
    

if __name__ == '__main__':
    main()

        


