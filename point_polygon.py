import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
import time
from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)
import random

DATA_PATH = "countrydata/Brazil.txt"

def getPolygonByData(dataPath):
    file1 = open(dataPath, "r")
    polygon = []
    
    for vertexLine in file1.readlines():
        polygon.append(tuple(int(coord) for coord in vertexLine.split(",")))

    return polygon

def edges(polygon):
    i = iter(polygon)
    first = prev = item = next(i)
    for item in i:
        yield prev, item
        prev = item
    yield item, first

def edgeOrientation(edge):
    return 1 if edge[1][1] > edge[0][1] else -1

def isPointInside(point, polygon):
    windingNumber = 0 
    for edge in edges(polygon):
        windingNumber += checkRayIntersection(point, edge)*edgeOrientation(edge)
    return windingNumber % 2 == 1

def checkRayIntersection(point, edge):
    if hitsVertex(point, edge):
        return 0.5
    elif isPointOutYpos(point, edge):
        return 0
    elif isPointLeftXpos(point, edge):
        return 1
    elif isPointLeftEdge(point, edge):
        return 1
    return 0

def hitsVertex(point, edge):
    if point[0] < edge[0][0] and point[1] == edge[0][1]:
        return True
    if point[0] < edge[1][0] and point[1] == edge[1][1]:
        return True
    return False

def isPointOutYpos(point, edge):
    return point[1] > max([edge[i][1] for i in range(2)]) or point[1] < min([edge[i][1] for i in range(2)])

def isPointLeftXpos(point, edge):
    return point[0] < min([edge[i][0] for i in range(2)])

def isPointLeftEdge(point, edge):
    if point[0] > max([edge[i][0] for i in range(2)]):
        return False

    arr = np.array([
        [1,1,1],
        [edge[0][0], edge[1][0], point[0]],
        [edge[0][1], edge[1][1], point[1]]
    ])

    det = np.linalg.det(arr)
    sign = 1 if edge[1][1] - edge[0][1] >= 0 else -1

    return det*sign > 0

def polygonDataMain():
    polygondata = getPolygonByData(DATA_PATH)
    
    xsizemax = max([polygondata[i][0] for i in range(len(polygondata))])
    ysizemax = max([polygondata[i][1] for i in range(len(polygondata))])

    xsize = 1000
    ysize = 1000

    polygon = [(x[0]//20000, x[1]//20000) for x in polygondata]

    im = Image.new('1', (xsize,ysize))

    for x in range(xsize):
        for y in range(ysize):
            if isPointInside((x,y), polygon):
                im.putpixel((x,y), ImageColor.getcolor('white', '1')) # or whatever color you wish

    im.save('simplePixel2.png') # or any image format

def simplePolygonMain():
    xsize = 500
    ysize = 500

    random.seed(5)
    polygon = random_convex_polygon(num_points=12)
    polygon.append(polygon[0])
    polygon = [(xsize*point[0], ysize*point[1]) for point in polygon]
    #polygon = [(300,100),(500,100),(680,300), (500,500), (300,500), (100,300)]
    #xsize, ysize, polygon = 10,10, [(0,0),(9,9),(9,0)]

    outpoints = [(1,1),(500,50),(450,500)]
    inpoints = [(200,200),(300,400),(100,400)]
    examplepoints = outpoints + inpoints

    plt.plot(*zip(*polygon), linewidth=3)
    plt.scatter(*zip(*examplepoints), marker = "*", color="orange")
    #plt.scatter(*zip(*outpoints), marker = "x", color="red")
    #plt.scatter(*zip(*inpoints), marker = ">", color="green")
    plt.show()

    im = Image.new('1', (xsize,ysize))

    for x in range(xsize):
        for y in range(ysize):
            if isPointInside((x,y), polygon):
                im.putpixel((x,y), ImageColor.getcolor('white', '1')) # or whatever color you wish

    im.save('simplePixel.png') # or any image format

def checkRandomPointsMain():
    xsize = 500
    ysize = 500
    numpoints = 200

    random.seed(5)
    polygon = random_convex_polygon(num_points=6)
    #polygon = getPolygonByData("countrydata/brazil.txt")
    maxxsize = max(polygon, key=lambda x : x[0])[0]
    maxysize = max(polygon, key=lambda x : x[1])[1]
    polygon.append(polygon[0])
    polygon = [(xsize*point[0], ysize*point[1]) for point in polygon]

    #polygon = [(xsize*point[0]//maxxsize, ysize - ysize*point[1]//maxysize) for point in polygon]
    #polygon = [(300,100),(500,100),(680,300), (500,500), (300,500), (100,300)]
    #xsize, ysize, polygon = 10,10, [(0,0),(9,9),(9,0)]

    examplepoints = [(i,j) for i in range(0, xsize, 10) for j in range(0, ysize, 10)]
    #examplepoints = [(random.uniform(0,xsize), random.uniform(0,ysize)) for i in range(numpoints)]
    inpoints = []
    outpoints = []

    start_time = time.time()
    for point in examplepoints:
        if isPointInside(point, polygon):
            inpoints.append(point)
        else:
            outpoints.append(point)
    print("--- {} seconds ---".format(time.time() - start_time))


    plt.plot(*zip(*polygon), linewidth=3)
    #plt.scatter(*zip(*examplepoints), marker = "*", color="orange")
    #plt.scatter(*zip(*outpoints), marker = "x", color="red")
    #plt.scatter(*zip(*inpoints), marker = ">", color="green")
    plt.show()

def timeanalysis():
    xsize = 500
    ysize = 500

    timesall = []

    gridrange = range(10,31,10)
    polyrange = range(5,102, 10)


    for gridpoint in gridrange:
        points = [(i, j) for i in np.linspace(0,xsize,gridpoint) for j in np.linspace(0,ysize,gridpoint)]
        times = []
        for numpoints in polyrange:
            random.seed(5)
            polygon = random_polygon(num_points=numpoints)
            
            

            elapsed = 0
            for k in range(20):
                inpoints = []
                outpoints = []

                start_time = time.time()
                for point in points:
                    if isPointInside(point, polygon):
                        inpoints.append(point)
                    else:
                        outpoints.append(point)
                elapsed += time.time() - start_time
            
            #dicttime = {"time": time.time() - start_time, "polygon": numpoints, "points": gridpoint*gridpoint}
            times.append(elapsed/20)
        timesall.append(times)


    legend = []
    for times in timesall:
        plt.plot(polyrange, times)

    plt.legend(["100 points", "400 points", "900 points"])
    plt.xlabel("Number of vertices of polygon")
    plt.ylabel("Time elapsed (sec)")
    plt.show()



if __name__ == "__main__":
    start_time = time.time()

    #simplePolygonMain()
    #polygonDataMain()
    #checkRandomPointsMain()
    timeanalysis()

    print("--- {} seconds ---".format(time.time() - start_time))


