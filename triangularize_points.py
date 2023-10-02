from orientation_utils import *
from functools import cmp_to_key
from convex_hull import graham
from utils import *
from point_polygon import isPointInside
import random
import time
import matplotlib.pyplot as plt

def triangularizepoints(points):
    # points are triangle
    if len(points) == 3:
        return [points]
    hull = graham(points)
    # points are convex hull
    if len(hull) == len(points) + 1:
        return [[points[0], points[i], points[i+1]] for i in range(1,len(points)-1)]

    # find a point inside the hull
    firstpoint = next((point for point in points if point not in hull))
    triangles = []

    trianglehulls = [[firstpoint,edge[0], edge[1]] for edge in edges(hull)]
    for trianglehull in trianglehulls:
        triangularize_rec(points, trianglehull, triangles)

def triangularize_rec(points):
    return

def triangularize_graham(points):
    leftmostPoint = min(points, key = lambda x : x[0])
    sortedPoints = sorted(points, key = cmp_to_key(lambda p1, p2 : -1 if p1[0] < p2[0] else (0 if p1[0] == p2[0] else 1) ))
    #sortedPoints = quicksort(points, leftmostPoint)

    return graham_triangularize_sorted(points, sortedPoints, leftmostPoint)

def graham_triangularize_sorted(points, sortedPoints, leftmostPoint):
    hull = []
    hull2 = []
    trianglesid = []
    adjlist = []

    # first step
    hull.append(0)
    hull.append(1)

    current = 2

    while current%len(points) != 0:
        while len(hull) > 1 and compare_ccw(sortedPoints[hull[-2]], sortedPoints[hull[-1]], sortedPoints[current]) != -1: # concave turn
            trianglesid.append([hull[-2], hull[-1], current])
            hull.pop()
        hull.append(current)
        current += 1

    # second step
    hull2.append(0)
    hull2.append(1)

    current = 2

    while current%len(points) != 0:
        while len(hull2) > 1 and compare_ccw(sortedPoints[hull2[-2]], sortedPoints[hull2[-1]], sortedPoints[current]) != 1: # concave turn
            trianglesid.append([hull2[-2], hull2[-1], current])
            hull2.pop()
        hull2.append(current)
        current += 1

    return [sortedPoints[id] for id in hull+hull2[-2::-1]],[[sortedPoints[t[0]],sortedPoints[t[1]],sortedPoints[t[2]]] for t in trianglesid]
    #return hull+hull2[-2::-1], triangles


def main():
    num_points = 25
    box_size = 100
    random.seed(1)
    points = [(box_size*random.random(), box_size*random.random()) for i in range(num_points)]
    
    print("--- {} points ---".format(len(points)))
    start_time = time.time()
    convexHull, triangles = triangularize_graham(points)
    print("--- {} milliseconds ---".format(1000*(time.time() - start_time)))
    
    plt.scatter(*zip(*points))
    plt.plot(*zip(*convexHull), marker='*', color="red")
    for triangle in triangles:
        plt.plot(*zip(*triangle), marker='*', color="green", linestyle="dashed")
    plt.show()

if __name__ == "__main__":
    main()