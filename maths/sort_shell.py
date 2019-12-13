#-*- coding:utf-8-*-
'''
shell sort:希尔排序，也称为增量递减排序。基本思路，是把原来的序列，等效视为一个矩阵的形式
'''
def shell_sort(collection):
    """Pure implementation of shell sort algorithm in Python
    :param collection:  Some mutable ordered collection with heterogeneous
    comparable items inside
    :return:  the same collection ordered by ascending

    >>> shell_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> shell_sort([])
    []

    >>> shell_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    # Marcin Ciura's gap sequence
    gaps = [701, 301, 132, 57, 23, 10, 4, 1]

    for gap in gaps:
        i = gap
        while i < len(collection):
            temp = collection[i]
            j = i
            while j >= gap and collection[j - gap] > temp:
                collection[j] = collection[j - gap]
                j -= gap
            collection[j] = temp
            i += 1

    return collection