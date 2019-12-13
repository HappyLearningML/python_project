#-*- coding:utf-8-*-
'''
bubble_sort冒泡排序：比较两个相邻的元素，将值大的元素交换至右端
'''

def bubble_sort(collection):
    """Pure implementation of bubble sort algorithm in Python

    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending

    Examples:
    >>> bubble_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> bubble_sort([])
    []

    >>> bubble_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    length = len(collection)
    for i in range(length):
        swapped = False
        for j in range(length-1):
            if collection[j] > collection[j+1]:
                swapped = True
                collection[j], collection[j+1] = collection[j+1], collection[j]
        if not swapped: break  # Stop iteration if the collection is sorted.
    return collection