import point_polygon
import numpy as np
from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)
import random
import matplotlib.pyplot as plt
import time
from utils import *
from orientation_utils import *

def triangleareasigned(triangle):
    arr = np.array([
    [1,1,1],
    [triangle[0][0], triangle[1][0], triangle[2][0]],
    [triangle[0][1], triangle[1][1], triangle[2][1]]
    ])

    return np.linalg.det(arr)

def triangularize(polygon):
    if len(polygon) == 3:
        return [polygon]
    
    leftmostpointid = np.argmin([point[0] for point in polygon])

    targetpoint = polygon[leftmostpointid]
    previouspoint = polygon[leftmostpointid-1]
    nextpoint = polygon[(leftmostpointid+1)%len(polygon)]

    triangle = [previouspoint, targetpoint, nextpoint]

    pointinsidefound = False
    cutpointid = -1
    cutpointarea = 0

    for pointid, point in enumerate(polygon):
        if pointid == leftmostpointid or pointid == (leftmostpointid+1)%len(polygon) or pointid == (leftmostpointid-1)%len(polygon):
            continue
        if point_polygon.isPointInside(point, triangle):
            pointinsidefound = True
            arr = np.array([
                [1,1,1],
                [nextpoint[0], previouspoint[0], point[0]],
                [nextpoint[1], previouspoint[1], point[1]]
                ])

            curcutpointarea = np.linalg.det(arr)
            if cutpointarea < curcutpointarea:
                cutpointarea = curcutpointarea
                cutpointid = pointid

    # if you didn't find a point inside 
    if cutpointid == -1:
        return [triangle] + triangularize([polygon[id] for id in range(len(polygon)) if id != leftmostpointid])
    
    len1 = (cutpointid - leftmostpointid)%len(polygon) + 1
    len2 = len(polygon) - len1 + 2
    polygon1 = [polygon[(id + leftmostpointid)%len(polygon)] for id in range(len1)]
    polygon2 = [polygon[(id + cutpointid)%len(polygon)] for id in range(len2)]
    #return triangularize(polygon[1:cutpointid+1]) + triangularize(polygon[cutpointid:]+[polygon[0], polygon[1]])
    return triangularize(polygon1) + triangularize(polygon2)

def assignColorToTriangle(triangleids, colorList):
    for i in range(3):
        for j in range(i,3):
            if colorList[triangleids[i]] == colorList[triangleids[j]] and colorList[triangleids[j]] != -1:
                print("failed to unique colors")
    # first is colored
    if colorList[triangleids[0]] != -1:
        # second is colored
        if colorList[triangleids[1]] != -1:
            # third not colored
            if colorList[triangleids[2]] == -1:
                colorList[triangleids[2]] = -1*(colorList[triangleids[0]]+colorList[triangleids[1]])%3 # 0,1 ; 1,2; 2,0
        # second not colored, third colored
        elif colorList[triangleids[2]] != -1:
            colorList[triangleids[1]] = -1*(colorList[triangleids[0]]+colorList[triangleids[2]])%3 # 0,1 ; 1,2; 2,0
        # second and third not colored
        else:
            colorList[triangleids[1]] = (colorList[triangleids[0]]+1)%3
            colorList[triangleids[2]] = (colorList[triangleids[1]]+1)%3
    # first not colored
    else:
        # second is colored
        if colorList[triangleids[1]] != -1:
            if colorList[triangleids[2]] != -1:
                colorList[triangleids[0]] = -1*(colorList[triangleids[1]]+colorList[triangleids[2]])%3
        # second not colored, third colored
        elif colorList[triangleids[2]] != -1:
            colorList[triangleids[0]] = (colorList[triangleids[2]]+1)%3
            colorList[triangleids[1]] = (colorList[triangleids[0]]+1)%3
        # no one is colored
        else:
            colorList[triangleids[0]] = 0
            colorList[triangleids[1]] = 1
            colorList[triangleids[2]] = 2

def triangularizenew(polygon):
    colorList = [-1 for i in range(len(polygon))]
    polygonids = [i for i in range(len(polygon))]
    return triangularizeiter(polygonids, colorList, polygon), colorList

def triangularizeiter(polygonids, colorList, polygon):
    if len(polygonids) < 3:
        print(polygonids)
        return []
    if len(polygonids) == 3:
        #assignColorToTriangle(polygonids, colorList)
        return [polygonids]
    
    #absoluteleftmostpointid = np.argmin([polygon[i][0] for i in polygonids]) # absolute id
    #xValueById = [polygon[polygonids[i]][0] for i in range(len(polygonids))]
    relativeleftmostpointid = np.argmin([polygon[polygonids[i]][0] for i in range(len(polygonids))])
    leftmostpointid = polygonids[relativeleftmostpointid] # absolute id

    targetid = leftmostpointid
    previousid = polygonids[relativeleftmostpointid-1]
    nextid = polygonids[(relativeleftmostpointid+1)%len(polygonids)]

    targetpoint = polygon[targetid]
    previouspoint = polygon[previousid]
    nextpoint = polygon[nextid]

    triangle = [previouspoint, targetpoint, nextpoint]
    triangleids = [previousid, targetid, nextid] # absolute id

    pointinsidefound = False
    cutpointid = -1
    cutpointarea = 0

    for relativeid, absoluteid in enumerate(polygonids):
        point = polygon[absoluteid]
        if absoluteid == targetid or absoluteid == previousid or absoluteid == nextid:
            continue
        if baricentricCoord(point, triangle):
        #if point_polygon.isPointInside(point, triangle):
            pointinsidefound = True

            curcutpointarea = determinant(nextpoint, previouspoint, point)
            if cutpointarea < curcutpointarea:
                cutpointarea = curcutpointarea
                cutpointid = relativeid

    # if you didn't find a point inside 
    if cutpointid == -1:
        #assignColorToTriangle(triangleids, colorList)
        return [triangleids] + triangularizeiter([id for id in polygonids if id != leftmostpointid], colorList, polygon)
    
    len1 = (cutpointid - relativeleftmostpointid)%len(polygonids) + 1
    len2 = len(polygonids) - len1 + 2
    polygon1 = [polygonids[(id + relativeleftmostpointid)%len(polygonids)] for id in range(len1)]
    polygon2 = [polygonids[(id + cutpointid)%len(polygonids)] for id in range(len2)]
    #return triangularize(polygon[1:cutpointid+1]) + triangularize(polygon[cutpointid:]+[polygon[0], polygon[1]])
    return triangularizeiter(polygon1, colorList, polygon) + triangularizeiter(polygon2, colorList, polygon)

def countNonColors(triangle, colorList):
    num = 0
    lastid = -1
    noncolor = 0
    for pointid in triangle:
        if colorList[pointid] == -1:
            num += 1
            lastid = pointid
        else:
            noncolor -= colorList[pointid]
    if num == 1:
        noncolor = noncolor%3
    else:
        noncolor = -1
    return num, lastid, noncolor 

def notAllColored(triangles, colorList):
    for triangle in triangles:
        num, lastid, noncolor = countNonColors(triangle,colorList)
        if num != 0:
            return True
    return False

def colorize(points, triangles, colorList):
    colorList[triangles[0][0]] = 0
    colorList[triangles[0][1]] = 1
    colorList[triangles[0][2]] = 2

    while notAllColored(triangles, colorList):
        for triangle in triangles:
            num, lastid, noncolor = countNonColors(triangle, colorList)
            if num == 1:
                colorList[lastid] = noncolor

def findLeastColor(colorList):
    count = [0,0,0]
    for color in colorList:
        count[color] += 1
    return np.argmin(count)

def timeanalysis():
    random.seed(5)
    num_turns = 20
    max_points = 500
    points_step = 5

    xaxis = [i for i in range(10,max_points,points_step)]
    timetotriangle = []
    timetocolorize = []

    for i in range(10,max_points,points_step):
        polygon = random_polygon(i)
        meantime1, meantime2 = 0,0
        for turn in range(num_turns):
            start_time = time.time()
            triangles, colorList = triangularizenew(polygon)
            meantime1 += time.time() - start_time

            start_time = time.time()
            colorize(polygon, triangles, colorList)
            meantime2 += time.time() - start_time
        meantime1 /= num_turns
        meantime2 /= num_turns
        timetotriangle.append(meantime1)
        timetocolorize.append(meantime2)

    plt.plot(xaxis, timetotriangle, color="green")
    plt.plot(xaxis, timetocolorize, color="red")
    plt.xlabel("Num Points")
    plt.ylabel("Time (sec)")
    plt.legend(["Triangularization", "Gallery of Art"])
    plt.show()


def main():
    num_points = 5
    random.seed(10)
    start_time = time.time()
    #polygon = random_convex_polygon(num_points=num_points)
    polygon = getPolygonByData(DATA_PATH)
    print("--- Generate",num_points,"polygon : {} seconds ---".format(time.time() - start_time))
    #polygon = [(1,1),(2,2),(3,1),(4,2),(5,1),(6,2),(1.5,2.5),(5.5,9),(4.5,8),(2.5,9)]
    #polygon = [(4,2),(5,1),(6,5),(3,9)]
    #polygon = [(3,1),(5,1),(7,3),(7,5),(5,5), (1, 3)]
    
    '''polygon = [(0.5, 2), (1,1.5), (1.5, 1.8), (2, 0.5), (2.5, 1.7),
               (2.5, 8-1.7),(2, 8-0.5),(1.5, 8-1.8),(1, 8-1.5), (0.5, 8-2),
               (0.9, 8-2.2), (1.3, 8-2.4),(1.9, 8-2.5), 
               (1.3, 2.4),(0.9, 2.2)]
    '''
    start_time = time.time()
    polygon = [(p[0]+0.1*random.random(), p[1]+0.1*random.random()) for p in polygon]
    print("--- Shuffle polygon: {} seconds ---".format(time.time() - start_time))
               #(0.4, 4), (1.1,3.5), (1.55, 3.3), (2.1, 4.5), (2.55, 4.2), (1.9, 2.5), (1.3, 2.4),(0.9, 2.2)]
   
    start_time = time.time()
    triangles, colorList = triangularizenew(polygon)
    print("--- Triangularize {} points: {} seconds ---".format(len(polygon),time.time() - start_time))

    start_time = time.time()
    colorize(polygon, triangles, colorList)
    print("--- Colorize: {} seconds ---".format(time.time() - start_time))

    cameracolor = findLeastColor(colorList)

    plt.plot(*zip(*(polygon+[polygon[0]])), linewidth=2, color = 'gray')

    for triangleid in triangles:
        triangle = [polygon[triangleid[0]],polygon[triangleid[1]],polygon[triangleid[2]]]
        colormap = {0 : 'red', 1 : 'orange', 2 : 'cyan', -1 : 'gray'}
        colors = [colormap[colorList[triangleid[i]]] for i in range(3)]
        #plt.plot(*zip(*(triangle+[triangle[0]])), linewidth=1, linestyle = 'dotted', color = 'green')
        #plt.scatter(*zip(*(triangle)), linewidth=1, color = colors, s = 52)

    mapchosencolor = lambda color : True if color == cameracolor else 15
    style = [polygon[i] for i in range(len(colorList)) if colorList[i] == cameracolor]
    plt.scatter(*zip(*style), linewidth=1, color=colormap[cameracolor], s = 52, marker="D")
    plt.show()




if __name__ == "__main__":
    #main()
    timeanalysis()