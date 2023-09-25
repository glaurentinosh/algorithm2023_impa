import numpy as np
import random
import matplotlib.pyplot as plt
import point_polygon
import time
from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)
import math
from functools import cmp_to_key

DATA_PATH = "countrydata/italy.txt"

def compare_ccw(point1, point2, origin):
    '''arr = np.array([
        [1,1,1],
        [point1[0], point2[0], origin[0]],
        [point1[1], point2[1], origin[1]]
    ])'''

    #det = np.linalg.det(arr)
    det = (point2[0]-point1[0])*(origin[1]-point1[1])-(origin[0]-point1[0])*(point2[1]-point1[1])
    if det == 0:
        dist1, dist2 = distance(point1, origin), distance(point2, origin)
        if dist1 == dist2:
            return 0
        return -1 if dist1 < dist2 else 1
    return 1 if det < 0 else -1
    
def distance(p1, p2):
    return ((p1[0] - p2[0]) * (p1[0] - p2[0]) +
            (p1[1] - p2[1]) * (p1[1] - p2[1]))

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

def graham(points):
    leftmostPoint = min(points, key = lambda x : x[0])
    #sortedPoints = sorted(points, key = cmp_to_key(lambda p1, p2 : compare_ccw(p1, p2, leftmostPoint)))
    #sortedPoints = bubblesort(points, leftmostPoint)
    sortedPoints = mergesort(points, leftmostPoint)

    return points+[points[0]] if len(points) < 4 else graham_sorted(points, sortedPoints, leftmostPoint)

def graham_sorted(points, sortedPoints, leftmostPoint):
    hull = []

    hull.append(sortedPoints[0])
    hull.append(sortedPoints[1])
    hull.append(sortedPoints[2])

    current = 3

    while current%len(points) != 0:
        while len(hull) > 1 and compare_ccw(hull[-2], hull[-1], sortedPoints[current]) != -1: # concave turn
            hull.pop()
        hull.append(sortedPoints[current])
        current += 1

    return hull+[hull[0]]

def jarvis(points):
    convexHullPoints = []
    leftmostPoint = min(points, key = lambda x : x[0])

    convexHullPoints.append(leftmostPoint)

    while len(convexHullPoints) < len(points):
        refpoint = convexHullPoints[-1]
        otherpoints = [point for point in points if point != refpoint]
        nextpoint = otherpoints[0]

        edge = (refpoint, nextpoint)
        
        for point in otherpoints[1:]:
            arr = np.array([
                [1,1,1],
                [edge[0][0], edge[1][0], point[0]],
                [edge[0][1], edge[1][1], point[1]]
            ])

            det = np.linalg.det(arr)
            if det < 0:
                nextpoint = point
                edge = (refpoint, nextpoint)
        
        convexHullPoints.append(nextpoint)
        if nextpoint == leftmostPoint:
            return convexHullPoints


def convexHull1(points):
    convexHullPoints = []

    for idx, point in enumerate(points):
        otherpoints = [point for idy,point in enumerate(points) if idy != idx]
        if isPointInSomeTriangle(point, otherpoints):
            convexHullPoints.append(point)
    
    return orderListOfPoints(convexHullPoints)

def orderListOfPointsOld(points):
    startpoint = points[0]
    orderedPoints = []
    orderedPoints.append(startpoint)
    otherpoints = points[1:]

    while len(otherpoints) > 0:
        refpoint = orderedPoints[-1]

        for i in range(len(otherpoints)):
            edge = (refpoint, otherpoints[i])
            if isEdgeInBorder(edge, otherpoints[i+1:]):
                orderedPoints.append(otherpoints[i])
                otherpoints = [x for x in otherpoints if x!=otherpoints[i]]
                break
    
    orderedPoints.append(startpoint)

    return orderedPoints

def orderListOfPoints(points):
    startpoint = points[0]
    orderedPoints = []
    orderedPoints.append(startpoint)
    otherpoints = points[1:]

    while len(otherpoints) > 0:
        refpoint = orderedPoints[-1]
        nextpoint = otherpoints[0]

        edge = (refpoint, nextpoint)
        
        for point in otherpoints[1:]:
            arr = np.array([
                [1,1,1],
                [edge[0][0], edge[1][0], point[0]],
                [edge[0][1], edge[1][1], point[1]]
            ])

            det = np.linalg.det(arr)
            if det < 0:
                nextpoint = point
                edge = (refpoint, nextpoint)
        
        orderedPoints.append(nextpoint)
        otherpoints = [point for point in otherpoints if point != nextpoint]

    orderedPoints.append(startpoint)
    return orderedPoints
            

def isEdgeInBorder(edge, points):
    for j in range(len(points)):
        arr = np.array([
            [1,1,1],
            [edge[0][0], edge[1][0], points[j][0]],
            [edge[0][1], edge[1][1], points[j][1]]
        ])

        det = np.linalg.det(arr)
        if det < 0:
            return False
    return True

def isEdgeInBorder2(edge, points):
    cursign = 0
    for j in range(len(points)):
        arr = np.array([
            [1,1,1],
            [edge[0][0], edge[1][0], points[j][0]],
            [edge[0][1], edge[1][1], points[j][1]]
        ])

        det = np.linalg.det(arr)
        if cursign == 0 and det != 0:
            cursign = 1 if det > 0 else -1
        elif det*cursign < 0:
            return False
    return True

def isPointInSomeTriangle(point, otherpoints):
    for i in range(len(otherpoints)):
        for j in range(i+1, len(otherpoints)):
            for k in range(j+1, len(otherpoints)):
                triangle = [otherpoints[i],otherpoints[j], otherpoints[k]]
                if point_polygon.isPointInside(point, triangle):
                    return False
                
    return True
                
    
def convexHull2(points):
    convexHullEdges = []
    convexHullPointsSet = set()

    for i in range(len(points)):
        for j in range(i+1, len(points)):
            edge = (points[i], points[j])
            if isEdgeInBorder2(edge, points):
                convexHullEdges.append(edge)
                convexHullPointsSet.add(edge[0])
                convexHullPointsSet.add(edge[1])

    convexHullPoints = list(convexHullPointsSet)
    #convexHullPoints = list(set([point for edge in convexHullEdges for point in edge]))
    #convexHullPoints = orderListOfEdges(convexHullEdges)

    return orderListOfPoints(convexHullPoints)

def orderListOfEdges(edges):
    otheredges = edges[1:]
    startedge = edges[0]
    curedge = startedge
    orderedPoints = []
    orderedPoints.append(curedge[0])
    orderedPoints.append(curedge[1])

    while(len(otheredges) > 1):
        for edge in otheredges:
            if edge[0] == curedge[1]:
                orderedPoints.append(edge[1])
                curedge = edge
                otheredges = [x for x in otheredges if x != edge]
            elif edge[1] == curedge[1]:
                orderedPoints.append(edge[0])
                curedge = (edge[1], edge[0])
                otheredges = [x for x in otheredges if x != edge]
    
    return orderedPoints
                

def checkPointOrientation(point, edge):
    arr = np.array([
        [1,1,1],
        [edge[0][0], edge[1][0], point[0]],
        [edge[0][1], edge[1][1], point[1]]
    ])

    det = np.linalg.det(arr)

    if det > 0: 
        return 1
    if det == 0:
        return 0
    if det < 0:
        return -1

def getPointsByData(datapath):
    file1 = open(datapath, "r")
    points = []
    
    for point in file1.readlines():
        points.append(tuple(float(coord) for coord in point.split(",")))

    return points

def plots():
    num_points = 1000
    box_size = 100
    random.seed(5)
    points = [(box_size*random.random(), box_size*random.random()) for i in range(num_points)]
    #points = getPointsByData(DATA_PATH)
    
    #points = [(3,0),(1,1),(0,3),(-1,1),(-3,0),(-1,-1),(0,-3),(1,-1)]
    #points = random_star_shaped_polygon(num_points=num_points)

    #points = [(1*box_size/3*random.random()*math.cos(4*math.pi*j/num_points),
    #            box_size*random.random()*math.sin(2*math.pi*j/num_points)) for j in range(num_points)]
    print("--- {} points ---".format(len(points)))

    start_time = time.time()
    convexHull = graham(points)
    print("--- {} milliseconds ---".format(1000*(time.time() - start_time)))
    
    plt.scatter(*zip(*points))
    plt.plot(*zip(*convexHull), marker='*', color="red")
    plt.show()

def compareTime():
    num_points = 100
    box_size = 100
    points = [(box_size*random.random(), box_size*random.random()) for i in range(num_points)]

    start_time = time.time()
    convexHull = convexHull1(points)
    print("--- {} seconds ---".format(time.time() - start_time))

    start_time = time.time()
    convexHull = convexHull2(points)
    print("--- {} seconds ---".format(time.time() - start_time))

    start_time = time.time()
    convexHull = jarvis(points)
    print("--- {} seconds ---".format(time.time() - start_time))

def alg1Time():
    box_size = 100
    num_turns = 20
    meantimes = []

    xaxis = range(20,1000,20)

    random.seed(5)
    #j = 0
    for num_points in xaxis:
        #print("points :", j)
        #j += 1
        meantime = 0
        for i in range(num_turns):
            #print("turn :", i)
            points = [(box_size*random.random(), box_size*random.random()) for i in range(num_points)]
            #points = random_polygon(num_points=num_points)
            start_time = time.time()
            convexHull = graham(points)
            meantime += time.time() - start_time

        meantime /= num_turns 
        meantimes.append(meantime)

    with open('plotdata/graham_merge.txt', 'w') as f:
        for line in list(zip(xaxis, meantimes)):
            f.write(f"{line}\n")
    

    plt.plot(xaxis, meantimes)
    plt.show()
    
def alg2Time():
    box_size = 100
    num_turns = 50
    meantimes = []

    xaxis = range(20,1000,20)

    random.seed(5)

    for num_points in xaxis:
        meantime = 0
        for i in range(num_turns):
            points = []
            for j in range(num_points):
                length = np.sqrt(np.random.uniform(0, 1))
                angle = np.pi * np.random.uniform(0, 2)

                x = length * np.cos(angle)
                y = length * np.sin(angle)

                points.append((x,y))

            start_time = time.time()
            convexHull = graham(points)
            meantime += time.time() - start_time

        meantime /= num_turns
        meantimes.append(meantime)

    with open('plotdata/graham_circle.txt', 'w') as f:
        for line in list(zip(xaxis, meantimes)):
            f.write(f"{line}\n")
    

    plt.plot(xaxis, meantimes)
    plt.show()

def openAndTreatFile(filepath):
    with open(filepath, "r") as f:
        lines = [eval(line) for line in f.readlines()]
        lines = list(zip(*lines))

    return lines

def logplots():
    line1 = openAndTreatFile("plotdata/graham.txt")
    line2 = openAndTreatFile("plotdata/graham_bubble.txt")
    line3 = openAndTreatFile("plotdata/graham_merge.txt")

    plt.plot(*line1)
    #plt.plot(*line2, color = 'orange')
    plt.plot(*line3, color = 'green')
    
    slope1, intercept1 = np.polyfit(np.log(line1[0][10:]), np.log(line1[1][10:]), 1) 
    slope2, intercept2 = np.polyfit(np.log(line2[0]), np.log(line2[1]), 1) 
    slope3, intercept3 = np.polyfit(np.log(line3[0]), np.log(line3[1]), 1) 

    regline1 = [intercept1 + slope1*x for x in np.log(line1[0][10:])]
    regline2 = [intercept2 + slope2*x for x in np.log(line2[0])]
    regline3 = [intercept3 + slope3*x for x in np.log(line3[0])]

    #plt.plot(line1[0][1:], regline1)
    #plt.plot(line2[0], regline2)
    #plt.plot(line3[0], regline3)

    #legend = ['Algorithm 1', 'Algorithm 2', 'Jarvis']+['slope1', 'slope2', 'slope3']
    legend = ['Built-in Sort', 'Bubble Sort', 'Merge Sort']
    #plt.legend(legend[2:3])
    plt.legend(legend[0:1]+legend[2:3])
    plt.xlabel("Num points")
    plt.ylabel("Time (sec)")

    print("1 :", slope1, ": 2 :", slope2, ": 3 :", slope3)

    plt.show()

def testgraham():
    num_points = 1000
    box_size = 100
    random.seed(1)
    points = [(box_size*random.random(), box_size*random.random()) for i in range(num_points)]
    leftmostPoint = min(points, key = lambda x : x[0])
    points = sorted(points, key = cmp_to_key(lambda p1, p2 : compare_ccw(p1, p2, leftmostPoint)))
    points = points[:3]
    print("--- {} points ---".format(len(points)))

    start_time = time.time()
    convexHull = graham(points)
    print("--- {} seconds ---".format(time.time() - start_time))
    
    plt.scatter(*zip(*points))
    plt.plot(*zip(*convexHull), marker='*', color="red")
    plt.show()


if __name__ == "__main__":
    #plots()
    #testgraham()
    #alg1Time()
    logplots()


