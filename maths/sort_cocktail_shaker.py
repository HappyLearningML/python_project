#-*- coding:utf-8-*-
'''
cocktail shaker sort:鸡尾酒排序是冒泡排序的轻微变形。
不同的地方在于，鸡尾酒排序是从低到高然后从高到低来回排序，而冒泡排序则仅从低到高去比较序列里的每个元素。
他可比冒泡排序的效率稍微好一点，原因是冒泡排序只从一个方向进行比对(由低到高)，每次循环只移动一个项目
'''
def cocktail_shaker_sort(unsorted):
    """
    Pure implementation of the cocktail shaker sort algorithm in Python.
    """
    for i in range(len(unsorted)-1, 0, -1):
        swapped = False
        
        for j in range(i, 0, -1):
            if unsorted[j] < unsorted[j-1]:
                unsorted[j], unsorted[j-1] = unsorted[j-1], unsorted[j]
                swapped = True

        for j in range(i):
            if unsorted[j] > unsorted[j+1]:
                unsorted[j], unsorted[j+1] = unsorted[j+1], unsorted[j]
                swapped = True
        
        if not swapped:
            return unsorted