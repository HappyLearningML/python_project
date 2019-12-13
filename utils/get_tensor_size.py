#-*- coding:utf-8 -*-
'''
获取tensor的大小，这是指在没有使用tensorflow的情况下
'''
from operator import mul

try:
    reduce
except NameError:
    from functools import reduce

def get_tensor_size(tensor):
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)