from orientation_utils import *
from functools import cmp_to_key
from convex_hull import graham
from utils import *
from point_polygon import isPointInside
import random
import time
import matplotlib.pyplot as plt
import numpy as np

def triangularizepoints(mypoints):
    points = mypoints.copy()
    # points are triangle
    if len(points) == 3:
        return [points]
    hull = graham(points)
    # points are convex hull
    if len(hull) == len(points) + 1:
        return [[points[0], points[i], points[i+1]] for i in range(1,len(points)-1)]

    # # find a point inside the hull
    points = [point for point in points if point not in hull]
    #firstpoint = next((point for point in points if point not in hull))
    #firstpoint = points[0]
    triangles = []

    #trianglehulls = [[firstpoint,edge[0], edge[1]] for edge in edges(hull)]
    trianglehulls = [[hull[0], hull[i], hull[i+1]] for i in range(1,len(hull)-1)]
    points = [point for point in points for h in trianglehulls if point not in h]

    while len(points) > 0:
        j = 0
        while j < len(trianglehulls) and len(points) > 0:
            if isPointInside(points[-1], trianglehulls[j]):    
                edges = [[trianglehulls[j][0],trianglehulls[j][1]],
                        [trianglehulls[j][1],trianglehulls[j][2]],
                        [trianglehulls[j][2],trianglehulls[j][0]]]
                hullsToAdd = [[points[-1],edge[0],edge[1]] for edge in edges]
                trianglehulls = trianglehulls[:j]+trianglehulls[j+1:]
                trianglehulls.append(hullsToAdd[0])
                trianglehulls.append(hullsToAdd[1])
                trianglehulls.append(hullsToAdd[2])
                points = points[:-1]
            j+=1
        points = points[:-1]

    return hull, trianglehulls

def triangularize_graham(points):
    leftmostPoint = min(points, key = lambda x : x[0])
    sortedPoints = sorted(points, key = cmp_to_key(lambda p1, p2 : -1 if p1[0] < p2[0] else (0 if p1[0] == p2[0] else 1) ))
    #sortedPoints = quicksort(points, leftmostPoint)

    return graham_triangularize_sorted(points, sortedPoints, leftmostPoint)

def graham_triangularize_sorted(points, sortedPoints, leftmostPoint):
    hull = []
    hull2 = []
    trianglesid = []
    triangle_num = 0
    hashmap = {}

    # first step
    hull.append(0)
    hull.append(1)

    current = 2

    while current%len(points) != 0:
        while len(hull) > 1 and compare_ccw(sortedPoints[hull[-2]], sortedPoints[hull[-1]], sortedPoints[current]) != -1: # concave turn
            thistriangle = [hull[-2], hull[-1], current,-1,-1,-1]
            trianglesid.append(thistriangle)
            updateHashmap(hashmap,triangle_num,thistriangle, trianglesid)
            triangle_num += 1
            hull.pop()
        hull.append(current)
        current += 1

    # second step
    hull2.append(0)
    hull2.append(1)

    current = 2

    while current%len(points) != 0:
        while len(hull2) > 1 and compare_ccw(sortedPoints[hull2[-2]], sortedPoints[hull2[-1]], sortedPoints[current]) != 1: # convex turn
            thistriangle = [hull2[-2], hull2[-1], current,-1,-1,-1]
            trianglesid.append(thistriangle)
            updateHashmap(hashmap,triangle_num,thistriangle, trianglesid)
            triangle_num += 1
            hull2.pop()
        hull2.append(current)
        current += 1

    for id in range(len(trianglesid)):
        updateHashmap(hashmap,id,trianglesid[id], trianglesid)

    return [
        sortedPoints[id] for id in hull+hull2[-2::-1]
        ], trianglesid, [
            [sortedPoints[t[0]],sortedPoints[t[1]],sortedPoints[t[2]]] for t in trianglesid]

def updateHashmap(hashmap, triangleid, triangle, trianglesid):
    if str(triangle[0:2]) not in hashmap:
        hashmap[str(triangle[0:2])] = triangleid
    else:
        triangle[5] = hashmap[str(triangle[0:2])] if triangle[5] == -1 else triangle[5]
        hashmap[str(triangle[0:2])] = triangleid
    if str(triangle[1:3]) not in hashmap:
        hashmap[str(triangle[1:3])] = triangleid
    else:
        triangle[3] = hashmap[str(triangle[1:3])] if triangle[3] == -1 else triangle[3]
        hashmap[str(triangle[1:3])] = triangleid
    if str(triangle[0:3:2]) not in hashmap:
        hashmap[str(triangle[0:3:2])] = triangleid
    else:
        triangle[4] = hashmap[str(triangle[0:3:2])] if triangle[4] == -1 else triangle[4]
        hashmap[str(triangle[0:3:2])] = triangleid


def checkCommonEdge(triangle1ids, triangle2ids):
    edge = []
    count = 0
    i, j = 0
    while i<3 and j<3 and count < 2:
        if triangle1ids[i] == triangle2ids[j]:
            count+=1
            edge.append(triangle1ids[i])
            i+=1
            j+=1
        elif triangle1ids[i] < triangle2ids[j]:
            i+=1
        else:
            j+=1
    return True, edge if count == 2 else False, []

def getCentroid(points):
    return (sum([p[0] for p in points])/len(points),sum([p[1] for p in points])/len(points))

def isInBorder(triangle, id):
    return triangle[3] in [-1,id] or triangle[4] in [-1,id] or triangle[5] in [-1,id]

def getNextTrianglePath(triangleid, adjList, path):
    for i in range(3,6):
        if adjList[triangleid][i] not in path:
            path.append(adjList[triangleid][i])
            break

def getPathFromCentroid(centroid, triangles, adjList):
    centertriangle = 0
    for i in range(len(triangles)):
        if isPointInside(centroid,triangles[i]):
            centertriangle = i
            break
    
    path = []
    path.append(centertriangle)

    while not isInBorder(adjList[centertriangle],centertriangle):
        getNextTrianglePath(centertriangle, adjList, path)
        centertriangle = path[-1]
    
    #path.append(adjList[centertriangle][3])

    return path

def plots():
    num_points = 11
    box_size = 100
    random.seed(1)
    #points = [(i+3*random.random(),j+3*random.random()) for i in range(5,200,19) for j in range(10,200,17)]
    points = sorted([(box_size*random.random(), box_size*random.random()) for i in range(num_points)])
    #points = getPolygonByData(DATA_PATH)
    centroid = (sum([p[0] for p in points])/len(points),sum([p[1] for p in points])/len(points))

    print("--- {} points ---".format(len(points)))
    start_time = time.time()
    convexHull, adjList, triangles = triangularize_graham(points)
    print("--- {} milliseconds ---".format(1000*(time.time() - start_time)))
    
    #print(adjList)

    pathFromCentroid = getPathFromCentroid(centroid, triangles, adjList)
    #print(pathFromCentroid[2],"path:", pathFromCentroid)
    trianglesInPath = [triangles[id] for id in pathFromCentroid]
    pathPoints = [centroid]+[getCentroid(triangle) for triangle in trianglesInPath]

    #print(*zip(*pathPoints))
    plt.plot(*zip(*pathPoints), marker='h', color="blue")

    plt.scatter(*zip(*points))
    plt.plot(*zip(*convexHull), marker='*', color="red")
    for id in range(len(triangles)):
        triangle = triangles[id]
        c = getCentroid(triangle)
        #plt.text(c[0],c[1],'{}'.format(id))
        plt.plot(*zip(*triangle), marker="x", markersize=1, color="green", linestyle="dotted", alpha=0.5)

    plt.show()
    
def generateTxtTime():
    box_size = 100
    num_turns = 20
    meantimes = []

    xaxis = range(20,2000,20)

    random.seed(5)
    j = 0
    for num_points in xaxis:
        print("points :", j)
        j += 1
        meantime = 0
        for i in range(num_turns):
            print("turn :", i)
            points = [(box_size*random.random(), box_size*random.random()) for i in range(num_points)]
            #points = random_polygon(num_points=num_points)
            start_time = time.time()
            hull, adjList, triangles = triangularize_graham(points)
            centroid = getCentroid(points)
            start_time = time.time()
            pathFromCentroid = getPathFromCentroid(centroid, triangles, adjList)
            #hull, triangles = triangularizepoints(points)
            meantime += time.time() - start_time

        meantime /= num_turns 
        meantimes.append(meantime)

    with open('plotdata/graham_path_test.txt', 'w') as f:
        for line in list(zip(xaxis, meantimes)):
            f.write(f"{line}\n")

    plt.plot(xaxis, meantimes)
    plt.show()

def logplots():
    line1 = openAndTreatFile("plotdata/graham_path.txt")
    line2 = openAndTreatFile("plotdata/triangularize_badly.txt")


    plt.loglog(*line1, color = 'purple')
    #plt.loglog(*line2, color = 'orange')
    #    
    slope1, intercept1 = np.polyfit(np.log(line1[0][10:]), np.log(line1[1][10:]), 1) 
    slope2, intercept2 = np.polyfit(np.log(line2[0][1:]), np.log(line2[1][1:]), 1) 

    regline1 = [intercept1 + slope1*x for x in np.log(line1[0][1:])]
    regline2 = [intercept2 + slope2*x for x in np.log(line2[0][1:])]

    legend = ['Centroid path']
    
    plt.legend(legend)
    plt.xlabel("Num points")
    plt.ylabel("Time (ms)")

    print("1 :", slope1, ": 2 :", slope2)

    plt.show()

def recursiveTriangularizeMain():
    num_points = 100
    box_size = 100
    random.seed(1)
    points = [(i+3*random.random(),j+3*random.random()) for i in range(5,200,19) for j in range(10,200,17)]
    #points = [(box_size*random.random(), box_size*random.random()) for i in range(num_points)]

    print("--- {} points ---".format(len(points)))
    start_time = time.time()
    convexHull, triangles = triangularizepoints(points)
    print("--- {} milliseconds ---".format(1000*(time.time() - start_time)))

    plt.scatter(*zip(*points))
    plt.plot(*zip(*convexHull), marker='*', color="red")
    for id in range(len(triangles)):
        triangle = triangles[id]
        c = getCentroid(triangle)
        #plt.text(c[0],c[1],'{}'.format(id))
        plt.plot(*zip(*triangle), marker="x", markersize=1, color="green", linestyle="dotted", alpha=0.5)

    plt.show()

if __name__ == "__main__":
    #plots()
    #generateTxtTime()
    logplots()
    #recursiveTriangularizeMain()