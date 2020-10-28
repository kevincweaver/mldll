from ctypes import c_void_p, c_double, c_int, cdll
from numpy.ctypeslib import ndpointer
import time

def timingdecorator(func):
    def functionwrapper(*args):
        time1 = time.time()
        print(func(*args))
        time2 = time.time()-time1
        print(func.__name__ + "() executed in: ", time2, "seconds")
    return functionwrapper

_mldll = cdll.LoadLibrary('MLDLL.dll')

@timingdecorator
def transp(matrixA):
    global _mldll
    _mldll.c_transp.restype = ndpointer(dtype=c_double, shape=(matrixA.shape[0],matrixA.shape[1]))
    result = _mldll.c_transp(c_void_p(matrixA.ctypes.data), c_int(matrixA.shape[0]), c_int(matrixA.shape[1]))
    return result

@timingdecorator
def transp2(matrixA):
    global _mldll
    _mldll.c_transp2.restype = ndpointer(dtype=c_double, shape=(matrixA.shape[0],matrixA.shape[1]))
    result = _mldll.c_transp2(c_void_p(matrixA.ctypes.data), c_int(matrixA.shape[0]), c_int(matrixA.shape[1]))
    return result

@timingdecorator
def mmult(matrixA,matrixB):
    global _mldll
    _mldll.c_mmult.restype = ndpointer(dtype=c_double, shape=(matrixA.shape[0],matrixB.shape[1]))
    result = _mldll.c_mmult(c_void_p(matrixA.ctypes.data), c_int(matrixA.shape[0]), c_int(matrixA.shape[1]),
                            c_void_p(matrixB.ctypes.data), c_int(matrixB.shape[0]), c_int(matrixB.shape[1]))
    return result

@timingdecorator
def mmult2(matrixA,matrixB):
    global _mldll
    _mldll.c_mmult2.restype = ndpointer(dtype=c_double, shape=(matrixA.shape[0],matrixB.shape[1]))
    result = _mldll.c_mmult2(c_void_p(matrixA.ctypes.data), c_int(matrixA.shape[0]), c_int(matrixA.shape[1]),
                             c_void_p(matrixB.ctypes.data), c_int(matrixB.shape[0]), c_int(matrixB.shape[1]))
    return result

@timingdecorator
def mmult3(matrixA,matrixB):
    global _mldll
    _mldll.c_mmult3.restype = ndpointer(dtype=c_double, shape=(matrixA.shape[0],matrixB.shape[1]))
    result = _mldll.c_mmult3(c_void_p(matrixA.ctypes.data), c_int(matrixA.shape[0]), c_int(matrixA.shape[1]),
                             c_void_p(matrixB.ctypes.data), c_int(matrixB.shape[0]), c_int(matrixB.shape[1]))
    return result

@timingdecorator
def mmult4(matrixA,matrixB):
    global _mldll
    _mldll.c_mmult4.restype = ndpointer(dtype=c_double, shape=(matrixA.shape[0],matrixB.shape[1]))
    result = _mldll.c_mmult4(c_void_p(matrixA.ctypes.data), c_int(matrixA.shape[0]), c_int(matrixA.shape[1]),
                             c_void_p(matrixB.ctypes.data), c_int(matrixB.shape[0]), c_int(matrixB.shape[1]))
    return result
