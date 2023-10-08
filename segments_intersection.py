from functools import cmp_to_key
from orientation_utils import compare_ccw
import random
from math import pi,cos,sin
import matplotlib.pyplot as plt
from trees import *

class Segment:
	def __init__(self, point1, point2):
		self.p1 = point1 if point1[0] <= point2[0] else point2
		self.p2 = point2 if self.p1 == point1 else point1

	def __getitem__(self,key):
		if key not in (0,1):
			raise KeyError
		return self.p1 if key == 0 else self.p2
	
	def getListPoints(self):
		return [self.p1, self.p2]
	
	def getCentroid(self):
		return [(self.p1[0] + self.p2[0])/2,(self.p1[1] + self.p2[1])/2 ]

def setSegmentByList(seglist : list[Segment]):
	return Segment(seglist[0],seglist[1])

def getEventList(segments : list[Segment]):
	L = len(segments)
	sortedList = [i for i in range(2*L)]
	cmpLambda = lambda i,j : -1 if segments[i%L][i//L][0] < segments[j%L][j//L][0] else (1 if segments[i%L][i//L][0] > segments[j%L][j//L][0] else 0)
	sortedList.sort(key=cmp_to_key(cmpLambda))
	
	return sortedList

def detectIntersection(segments : list[Segment]):
    eventList = getEventList(segments)
    state = AVL_Tree()
    root = None
    L = len(segments)

    for i in eventList:
        if i // L == 0:
            root = state.insert(root, i, segments[i%L][i//L][1])
        elif i // L == 1:
            root = state.delete(root, i, segments[i%L][i//L][1])

        if state.getHeight(root) > 1:
            firstNode = state.getMinValueNode(root)
            while(state.getNextNode(firstNode) is not None):
                id1 = state.id
                id2 = state.getNextNode(firstNode).id
                if checkSegmentIntersection(segments[id1], segments[id2]) == True:
                    return True

    return False

def checkSegmentIntersection(segment1, segment2):
    ccw1 = compare_ccw(segment1[0], segment1[1], segment2[0])*compare_ccw(segment1[0], segment1[1], segment2[1])
    ccw2 = compare_ccw(segment2[0], segment2[1], segment1[0])*compare_ccw(segment2[0], segment2[1], segment1[1])

    if ccw1 == 0 or ccw2 == 0:
        return ccw1 < 0 or ccw2 < 0
    return ccw1 < 0 and ccw2 < 0

def checkSegmentsIntersectionTrivial(segments):
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            if checkSegmentIntersection(segments[i], segments[j]):
                return True
    return False

def checkSegmentsIntersection(segments):
    events = [i for i in range(len(segments)) for j in range(2)]
    events.sort(key = cmp_to_key(lambda i,j : -1 if segments[i][0] < segments[j][0] else 1))

def generateRandomSegment(boxsize, maxlength):
    segment = []

    radius = random.random()*maxlength
    angle = random.random()*2*pi
    displacement = (radius*cos(angle), radius*sin(angle))

    center = (random.random()*boxsize, random.random()*boxsize)

    for i in range(2):
        segment.append(tuple(map(sum,zip(center,displacement))))
        displacement = tuple(map(lambda i : -i, displacement))

    return segment

def generateRandomSegmentBox(xmin, xmax, ymin, ymax):
    segment = []

    maxlength = min(xmax - xmin, ymax - ymin)/2

    radius = random.random()*maxlength
    angle = random.random()*2*pi
    displacement = (radius*cos(angle), radius*sin(angle))

    #center = (xmin + random.random()*(xmax - xmin), ymin + random.random()*(ymax - ymin))
    center = (xmin + (xmax - xmin)/2, ymin + (ymax - ymin)/2)
              
    for i in range(2):
        segment.append(tuple(map(sum,zip(center,displacement))))
        displacement = tuple(map(lambda i : -i, displacement))

    return segment

def plotsegmentsMain():
    num_segments = 100
    box_size = 100
    max_length = box_size*0.3
    cell_size = 20
    random.seed(1)
    segments = [generateRandomSegment(box_size,max_length) for i in range(num_segments)]
    #segments = [generateRandomSegmentBox(i,i+cell_size,j,j+cell_size) for i in range(0,box_size,cell_size) for j in range(0,box_size,cell_size)]

    for segment in segments:
        plt.plot(*zip(*segment), color = 'green', marker='s')

    plt.show()

    print(checkSegmentsIntersectionTrivial(segments))

def testSegmentClass():
    segment = Segment((1,2),(3,4))
    print(segment.getListPoints())

    print(segment[0])
    print(segment[1])
    
    box_size = 100
    max_length = box_size*0.3
    num_segments = 10
    random.seed(1)

    segments = [setSegmentByList(generateRandomSegment(box_size, max_length)) for i in range(num_segments)]
    print([i%len(segments) for i in getEventList(segments)])

    for id,segment in enumerate(segments):
        s = segment.getListPoints()
        c = segment.getCentroid()
        plt.plot(*zip(*s), color = 'green', marker='s')
        plt.text(c[0],c[1],'{}'.format(id))

    plt.show()

def checkIntersectionMain():
    
    return

if __name__ == "__main__":
    plotsegmentsMain()
    #testSegmentClass()
    #checkIntersectionMain()