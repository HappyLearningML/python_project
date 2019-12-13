#-*- coding:utf-8-*-
'''
bucket sort:
桶排序(BucketSort) 小结：

1 桶排序核心思想是：根据数据规模n划分，m个相同大小的区间 （每个区间为一个桶，桶可理解为容器）

2 每个桶存储区间内的元素(区间为半开区间例如[0,10)或者[200,300) )

3 将n个元素按照规定范围分布到各个桶中去

4 对每个桶中的元素进行排序，排序方法可根据需要，选择快速排序，或者归并排序，或者插入排序

5 依次从每个桶中取出元素，按顺序放入到最初的输出序列中(相当于把所有的桶中的元素合并到一起)

6 桶可以通过数据结构链表实现

7 基于一个前提，待排序的n个元素大小介于0~k之间的整数 或者是(0, 1)的浮点数也可（算法导论8.4的例子） 

8 桶排序的时间代价，假设有m个桶，则每个桶的元素为n/m;

当辅助函数为冒泡排序O(n2)时,桶排序为 O(n)+mO((n/m)2);

当辅助函数为快速排序时O(nlgn)时,桶排序为 O(n)+m*O(n/m log(n/m))

9 通常桶越多，执行效率越快，即省时间，但是桶越多，空间消耗就越大，是一种通过空间换时间的方式
'''
import sort_insertion
import math

DEFAULT_BUCKET_SIZE = 5

def bucketSort(myList, bucketSize=DEFAULT_BUCKET_SIZE):
    if(len(myList) == 0):
        print('You don\'t have any elements in array!')

    minValue = myList[0]
    maxValue = myList[0]

    # For finding minimum and maximum values
    for i in range(0, len(myList)):
        if myList[i] < minValue:
            minValue = myList[i]
        elif myList[i] > maxValue:
            maxValue = myList[i]

    # Initialize buckets
    bucketCount = math.floor((maxValue - minValue) / bucketSize) + 1
    buckets = []
    for i in range(0, bucketCount):
        buckets.append([])

    # For putting values in buckets
    for i in range(0, len(myList)):
        buckets[math.floor((myList[i] - minValue) / bucketSize)].append(myList[i])

    # Sort buckets and place back into input array
    sortedArray = []
    for i in range(0, len(buckets)):
        sort_insertion.insertion_sort(buckets[i])
        for j in range(0, len(buckets[i])):
            sortedArray.append(buckets[i][j])

    return sortedArray

if __name__ == '__main__':
    sortedArray = bucketSort([12, 23, 4, 5, 3, 2, 12, 81, 56, 95])
    print(sortedArray)