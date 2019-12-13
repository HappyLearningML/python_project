#-*- coding:utf-8-*-
'''
pancake sort:煎饼排序,对一堆无序的煎饼以大小排序，铲子可以在任意位置伸进去并且把上面的煎饼都翻转过来
'''

def pancakesort(arr):
    cur = len(arr)
    while cur > 1:
        # Find the maximum number in arr
        mi = arr.index(max(arr[0:cur]))
        # Reverse from 0 to mi 
        arr = arr[mi::-1] + arr[mi+1:len(arr)]
        # Reverse whole list 
        arr = arr[cur-1::-1] + arr[cur:len(arr)]
        cur -= 1
    return arr