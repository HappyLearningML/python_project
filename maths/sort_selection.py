#-*- coding:utf-8-*-
'''
selection_sort:选择排序，就是每次找到未排序中最小的（最大的也行）元素的位置，找到后与该位置与未排序序列的第一个元素交换值，直到该序列成为有序序列
'''

def selection_sort(collection):
    """Pure implementation of the selection sort algorithm in Python
    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending


    Examples:
    >>> selection_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> selection_sort([])
    []

    >>> selection_sort([-2, -5, -45])
    [-45, -5, -2]
    """

    length = len(collection)
    for i in range(length):
        least = i
        for k in range(i + 1, length):
            if collection[k] < collection[least]:
                least = k
        collection[least], collection[i] = (
            collection[i], collection[least]
        )
    return collection