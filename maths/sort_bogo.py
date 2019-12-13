#-*- coding:utf-8-*-
'''
bogo排序法
Bogo排序算是一个恶搞排序，它将一切交给了上帝。Bogo排序每一趟都在检测待排序序列是否有序，如果有序，结束。如果无序，将待排序序列随机打乱（随机打乱！！！！！），重新检测
'''
import random

def bogosort(collection):
    """Pure implementation of the bogosort algorithm in Python
    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending
    Examples:
    >>> bogosort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> bogosort([])
    []
    >>> bogosort([-2, -5, -45])
    [-45, -5, -2]
    """

    def isSorted(collection):
        if len(collection) < 2:
            return True
        for i in range(len(collection) - 1):
            if collection[i] > collection[i + 1]:
                return False
        return True

    while not isSorted(collection):
        random.shuffle(collection)
    return collection

if __name__ == '__main__':
    print(bogosort([0,5,3,2,2]))