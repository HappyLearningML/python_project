#-*- coding:utf-8-*-
'''
merge sort:归并排序，将待排序数据分成两部分，继续将两个子部分进行递归的归并排序；然后将已经有序的两个子部分进行合并，最终完成排序
'''
def merge_sort(collection):
    """Pure implementation of the merge sort algorithm in Python

    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending

    Examples:
    >>> merge_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> merge_sort([])
    []

    >>> merge_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    length = len(collection)
    if length > 1:
        midpoint = length // 2
        left_half = merge_sort(collection[:midpoint])
        right_half = merge_sort(collection[midpoint:])
        i = 0
        j = 0
        k = 0
        left_length = len(left_half)
        right_length = len(right_half)
        while i < left_length and j < right_length:
            if left_half[i] < right_half[j]:
                collection[k] = left_half[i]
                i += 1
            else:
                collection[k] = right_half[j]
                j += 1
            k += 1

        while i < left_length:
            collection[k] = left_half[i]
            i += 1
            k += 1

        while j < right_length:
            collection[k] = right_half[j]
            j += 1
            k += 1

    return collection

def merge_sort_fast(LIST):
    start = []
    end = []
    while len(LIST) > 1:
        a = min(LIST)
        b = max(LIST)
        start.append(a)
        end.append(b)
        LIST.remove(a)
        LIST.remove(b)
    if LIST: start.append(LIST[0])
    end.reverse()
    return (start + end)
