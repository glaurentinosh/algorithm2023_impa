DATA_PATH = "countrydata/guam.txt"

def getPolygonByData(dataPath):
    file1 = open(dataPath, "r")
    polygon = []
    
    for vertexLine in file1.readlines():
        polygon.append(tuple(float(coord) for coord in vertexLine.split(",")))

    return polygon

def openAndTreatFile(filepath):
    with open(filepath, "r") as f:
        lines = [eval(line) for line in f.readlines()]
        lines = list(zip(*lines))

    return lines

def edges(polygon):
    i = iter(polygon)
    first = prev = item = next(i)
    for item in i:
        yield prev, item
        prev = item
    yield item, first