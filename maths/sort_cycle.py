#-*- coding:utf-8-*-
'''
cycle sort:圈排序，圈排序只给那些需要计数的数字计数
'''
def cycle_sort(array):
    ans = 0

    # Pass through the array to find cycles to rotate.
    for cycleStart in range(0, len(array) - 1):
        item = array[cycleStart]

        # finding the position for putting the item.
        pos = cycleStart
        for i in range(cycleStart + 1, len(array)):
            if array[i] < item:
                pos += 1

        # If the item is already present-not a cycle.
        if pos == cycleStart:
            continue

        # Otherwise, put the item there or right after any duplicates.
        while item == array[pos]:
            pos += 1
        array[pos], item = item, array[pos]
        ans += 1

        # Rotate the rest of the cycle.
        while pos != cycleStart:

            # Find where to put the item.
            pos = cycleStart
            for i in range(cycleStart + 1, len(array)):
                if array[i] < item:
                    pos += 1

            # Put the item there or right after any duplicates.
            while item == array[pos]:
                pos += 1
            array[pos], item = item, array[pos]
            ans += 1

    return ans