DATA_PATH = "countrydata/nazcamonkey.txt"

def getPolygonByData(dataPath):
    file1 = open(dataPath, "r")
    polygon = []
    
    for vertexLine in file1.readlines():
        polygon.append(tuple(float(coord) for coord in vertexLine.split(",")))

    return polygon