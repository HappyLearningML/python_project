#-*- coding:utf-8-*-
'''
基数排序：透过键值的部份资讯，将要排序的元素分配至某些“桶”中，藉以达到排序的作用
'''

def radixsort(lst):
  RADIX = 10
  maxLength = False
  tmp , placement = -1, 1

  while not maxLength:
    maxLength = True
    # declare and initialize buckets
    buckets = [list() for _ in range( RADIX )]

    # split lst between lists
    for i in lst:
      tmp = int((i / placement) % RADIX)
      buckets[tmp].append(i)

      if maxLength and tmp > 0:
        maxLength = False

    # empty lists into lst array
    a = 0
    for b in range( RADIX ):
      buck = buckets[b]
      for i in buck:
        lst[a] = i
        a += 1

    # move to next
    placement *= RADIX