from ctypes import c_void_p, c_double, c_int, cdll
from numpy.ctypeslib import ndpointer
import time
import numpy as np

mydll = cdll.LoadLibrary('MLDLL.dll')

matrix = np.full((1000,1000), 1, dtype=np.float64)             
n = matrix.shape[0]
m = matrix.shape[1]

matrix2 = np.full((1000,1000), 1, dtype=np.float64)
n2 = matrix2.shape[0]
m2 = matrix2.shape[1]

time1 = time.time()
transp = mydll.c_transp
transp.restype = ndpointer(dtype=c_double, shape=(m,n))
result = transp(c_void_p(matrix.ctypes.data), c_int(n), c_int(m))
print(result)
time2 = time.time()-time1
print("running time in seconds:", time2)

time1 = time.time()
transp2 = mydll.c_transp2
transp2.restype = ndpointer(dtype=c_double, shape=(m,n))
result = transp(c_void_p(matrix.ctypes.data), c_int(n), c_int(m))
print(result)
time2 = time.time()-time1
print("running time in seconds:", time2)

time1 = time.time()
mmult = mydll.c_mmult
mmult.restype = ndpointer(dtype=c_double, shape=(n,m2))
result = mmult(c_void_p(matrix.ctypes.data), c_int(n), c_int(m),
               c_void_p(matrix2.ctypes.data), c_int(n2), c_int(m2))
print(result)
time2 = time.time()-time1
print("running time in seconds:", time2)

time1 = time.time()
mmult2 = mydll.c_mmult2
mmult2.restype = ndpointer(dtype=c_double, shape=(n,m2))
result = mmult2(c_void_p(matrix.ctypes.data), c_int(n), c_int(m),
                c_void_p(matrix2.ctypes.data), c_int(n2), c_int(m2))
print(result)
time2 = time.time()-time1
print("running time in seconds:", time2)

time1 = time.time()
mmult3 = mydll.c_mmult3
mmult3.restype = ndpointer(dtype=c_double, shape=(n,m2))
result = mmult3(c_void_p(matrix.ctypes.data), c_int(n), c_int(m),
                c_void_p(matrix2.ctypes.data), c_int(n2), c_int(m2))
print(result)
time2 = time.time()-time1
print("running time in seconds:", time2)
