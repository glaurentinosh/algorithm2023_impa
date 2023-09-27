import random
from orientation_utils import *

def quicksort(alist, origin):
    newlist = alist.copy()
    random.shuffle(newlist)
    start = 0
    end = len(newlist)-1
    quickconquer(newlist, start, end, origin)
    return newlist

def getPivotPos(start, end):
    return random.randrange(start, end+1)

def quickdivide(alist, start, end, origin):
    pivot = alist[end]
    
    pointer = start

    for j in range(start, end):
        
        if compare_ccw(alist[j], pivot, origin) == -1: #alist[j] < pivot:
            alist[pointer], alist[j] = alist[j], alist[pointer]
            pointer += 1
        
    alist[pointer], alist[end] = alist[end], alist[pointer]
    return pointer

def quickconquer(alist, start, end, origin):
    if start < end:
        pivot = quickdivide(alist, start, end, origin)
        quickconquer(alist, start, pivot-1, origin)
        quickconquer(alist, pivot+1, end, origin)

def bubblesort(sortingpoints, origin):
    points = sortingpoints.copy()
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            if compare_ccw(points[i], points[j], origin) == 1:
                points[i], points[j] = points[j], points[i]
    return points

def mergeconquer(points, origin, start, medium, end):
    size1 = medium - start + 1
    size2 = end - medium

    pointer = start
    pointer1 = 0
    pointer2 = 0

    arr1 = [points[start + i] for i in range(size1)]
    arr2 = [points[medium + 1 + i] for i in range(size2)]

    while pointer1 < size1 and pointer2 < size2:
        if compare_ccw(arr1[pointer1], arr2[pointer2], origin) == 1:
            points[pointer] = arr2[pointer2]
            pointer2 += 1
        else:
            points[pointer] = arr1[pointer1]
            pointer1 += 1
        pointer += 1
    
    while pointer1 < size1:
        points[pointer] = arr1[pointer1]
        pointer1 += 1
        pointer += 1

    while pointer2 < size2:
        points[pointer] = arr2[pointer2]
        pointer2 += 1
        pointer += 1

def mergedivide(points, origin, start, end):
    if start < end:
        medium = start + (end - start)//2
        mergedivide(points, origin, start, medium)
        mergedivide(points, origin, medium+1, end)
        mergeconquer(points, origin, start, medium, end)

def mergesort(sortingpoints, origin):
    points = sortingpoints.copy()
    mergedivide(points, origin, 0, len(points)-1)
    return points


if __name__ == '__main__':
    alist = [random.random() for i in range(10)]
    sortlist = quicksort(alist)

    for e in sortlist:
        print(e)