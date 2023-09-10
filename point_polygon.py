import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
import time

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
    xsize = 1000
    ysize = 800

    #polygon = [(300,100),(500,100),(680,300), (500,500), (300,500), (100,300)]
    xsize, ysize, polygon = 10,10, [(0,0),(9,9),(9,0)]

    im = Image.new('1', (xsize,ysize))

    for x in range(xsize):
        for y in range(ysize):
            if isPointInside((x,y), polygon):
                im.putpixel((x,y), ImageColor.getcolor('white', '1')) # or whatever color you wish

    im.save('simplePixel.png') # or any image format

if __name__ == "__main__":
    start_time = time.time()

    #simplePolygonMain()
    polygonDataMain()

    print("--- {} seconds ---".format(time.time() - start_time))


