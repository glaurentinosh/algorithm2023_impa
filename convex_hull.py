import numpy as np
import random
import matplotlib.pyplot as plt
import point_polygon
import time

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


def plots():
    num_points = 100
    box_size = 100
    points = [(box_size*random.random(), box_size*random.random()) for i in range(num_points)]

    convexHull = convexHull2(points)
    
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
    meantimes = []

    xaxis = range(5,100,5)

    for num_points in xaxis:
        meantime = 0
        for i in range(10):
            points = [(box_size*random.random(), box_size*random.random()) for i in range(num_points)]
            start_time = time.time()
            convexHull = convexHull1(points)
            meantime += time.time() - start_time

        meantime /= 10 
        meantimes.append(meantime)

    plt.plot(xaxis, meantimes)
    plt.show()

if __name__ == "__main__":
    plots()


