import struct
import numpy as np
from functools import reduce
import os

"""
Tensorflow.Tensor -> sess.run -> numpy.array
RbRuntime -> std::vector<float> -> .bin
"""

def check_bin(path:str,shape):
    len = reduce(lambda x,y:x*y, np.array(shape))
    print("Size of .bin file: {}".format(os.path.getsize(path)))
    print("Size of float array with shape {} should be: {}".format(shape,len * struct.calcsize("f")))
    assert(os.path.getsize(path) == len * struct.calcsize("f"))

def decode(path,shape):
    len = reduce(lambda x,y:x*y, np.array(shape))
    file = open(path,"rb")
    array = np.array([struct.unpack("@f",file.read(struct.calcsize('f'))) for _ in range(len)])
    return array.resize(shape)


def diff_abs_mean(array1,array2) -> float :
    array1 = np.array(array1)
    array2 = np.array(array2)
    return float(np.mean(np.abs(array1-array2)))


def diff_abs_max(array1,array2) -> float :
    array1 = np.array(array1)
    array2 = np.array(array2)
    return float(np.max(np.abs(array1-array2)))


def diff_argmax(array1,array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    return np.argmax(np.abs(array1-array2))













