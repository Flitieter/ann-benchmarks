import numpy as np
import ctypes

my_functions = ctypes.CDLL('../build/libhybrid_search.so')

my_functions.py_interface.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]

class MyClass:
    def fit(self, X):
        rows, cols = X.shape
        X_flat = X.flatten()
        print("rows", rows)
        print("cols", cols)
        print(X_flat)
        X_ctypes = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        my_functions.py_interface(X_ctypes, rows, cols)

# 示例使用
my_class = MyClass()
#[data1[d1,d2,...,dm],data2[d1,d2,...,dm]]
X = np.array([[1.0, 2.0, 6.666], [3.0, 4.0, 8.888]], dtype=np.float32)
print("Before:", X)
my_class.fit(X);
# X_transformed = my_class.fit(X)
