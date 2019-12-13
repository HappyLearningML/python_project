#-*- coding:utf-8-*-
'''
gnome_sort:号称最简单的排序算法,只有一层循环,默认情况下前进冒泡,一旦遇到冒泡的情况发生就往回冒,直到把这个数字放好为止
'''
def gnome_sort(unsorted):
    """
    Pure implementation of the gnome sort algorithm in Python.
    """
    if len(unsorted) <= 1:
        return unsorted
        
    i = 1
    
    while i < len(unsorted):
        if unsorted[i-1] <= unsorted[i]:
            i += 1
        else:
            unsorted[i-1], unsorted[i] = unsorted[i], unsorted[i-1]
            i -= 1
            if (i == 0):
                i = 1