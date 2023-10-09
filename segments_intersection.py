from functools import cmp_to_key
import math
from orientation_utils import compare_ccw
import random
from math import pi,cos,sin
import matplotlib.pyplot as plt
from trees import *
import numpy as np
import time

class Segment:
	def __init__(self, point1, point2):
		self.p1 = point1 if point1[0] <= point2[0] else point2
		self.p2 = point2 if self.p1 == point1 else point1

	@classmethod
	def from_list(cls, segmentList):
		return cls(*segmentList)

	def __getitem__(self,key):
		if key not in (0,1):
			raise IndexError
		return self.p1 if key == 0 else self.p2
	 
	def __setitem__(self,key,value):
		if key == 0:
			self.p1 = value
		elif key == 1:
			self.p2 = value
		else:
			raise IndexError
	
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

def detectIntersectionList(segments : list[Segment]):
	eventList = getEventList(segments)
	state = []

	L = len(segments)

	for i in eventList:
		if i // L == 0:
			state.append((i, segments[i%L][i//L][1]))
			curNode = state[-1]
			nextNode = state.getNextNode(curNode)
			prevNode = state.getPreviousNode(curNode)
			id1 = prevNode.id if prevNode is not None else None
			id2 = nextNode.id if nextNode is not None else None
			
			if id1 is None or id2 is None:
				continue

			if checkSegmentIntersection(segments[id1], segments[i]) == True:
				return True, id1, i
			if checkSegmentIntersection(segments[id2], segments[i]) == True:
				return True, id2, i
			
		elif i // L == 1:
			curNode = state.getNodeById(root,i,segments[i%L][i//L][1])
			nextNode = state.getNextNode(curNode)
			prevNode = state.getPreviousNode(curNode)
			id1 = prevNode.id if prevNode is not None else None
			id2 = nextNode.id if nextNode is not None else None
			
			if id1 is None or id2 is None:
				continue

			if checkSegmentIntersection(segments[id1], segments[id2]) == True:
				return True, id1, id2

			root = state.delete(root, i, segments[i%L][i//L][1])

	return False, None, None


def detectIntersection(segments : list[Segment]):
	eventList = getEventList(segments)
	stateTree = AVL_Tree()
	root = None
	L = len(segments)

	for i in eventList:
		if i // L == 0:
			root = stateTree.insert(root, i, segments[i%L][i//L][1])

			curNode = stateTree.getNodeById(root,i,segments[i%L][i//L][1])
			nextNode = stateTree.getNextNode(curNode)
			prevNode = stateTree.getPreviousNode(curNode)
			id1 = prevNode.id if prevNode is not None else None
			id2 = nextNode.id if nextNode is not None else None
			
			if id1 is None or id2 is None:
				continue

			if checkSegmentIntersection(segments[id1], segments[i]) == True:
				return True, id1, i
			if checkSegmentIntersection(segments[id2], segments[i]) == True:
				return True, id2, i
			
		elif i // L == 1:
			curNode = stateTree.getNodeById(root,i,segments[i%L][i//L][1])
			nextNode = stateTree.getNextNode(curNode)
			prevNode = stateTree.getPreviousNode(curNode)
			id1 = prevNode.id if prevNode is not None else None
			id2 = nextNode.id if nextNode is not None else None
			
			if id1 is None or id2 is None:
				continue

			if checkSegmentIntersection(segments[id1], segments[id2]) == True:
				return True, id1, id2

			root = stateTree.delete(root, i, segments[i%L][i//L][1])

	return False, None, None

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

def generateRandomSegmentRangeLen(boxsize, length, randlength = 0):
	segment = []

	radius = (length + np.random.normal()*randlength)/2
	angle = random.random()*2*pi
	displacement = (radius*cos(angle), radius*sin(angle))

	center = (random.random()*boxsize, random.random()*boxsize)

	for i in range(2):
		segment.append(tuple(map(sum,zip(center,displacement))))
		displacement = tuple(map(lambda i : -i, displacement))

	return segment#Segment.from_list(segment)#setSegmentByList(segment)

# def generateRandomSegmentRangeLenBox(xmin, xmax, ymin, ymax, length, randlength = 0):
# 	segment = []

# 	radius = (length + np.random.normal()*randlength)/2
# 	angle = random.random()*2*pi
# 	displacement = (radius*cos(angle), radius*sin(angle))

# 	center = ((xmax-xmin)/2, (ymax - ymin)/2)

# 	for i in range(2):
# 		segment.append(tuple(map(sum,zip(center,displacement))))
# 		displacement = tuple(map(lambda i : -i, displacement))

# 	return segment#Segment.from_list(segment)#setSegmentByList(segment)


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
	num_segments = 1000
	box_size = 1000
	length = box_size*0.01
	varlength = box_size*0.01
	cell_size = 20
	#random.seed(1)
	#segments = [generateRandomSegment(box_size,max_length) for i in range(num_segments)]
	#segments = [generateRandomSegmentBox(i,i+cell_size,j,j+cell_size) for i in range(0,box_size,cell_size) for j in range(0,box_size,cell_size)]
	segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length,varlength)) for i in range(num_segments)]

	#print(checkSegmentsIntersectionTrivial(segments))
	detected, id1, id2 = detectIntersection(segments)
	print(detected)

	for id,segment in enumerate(segments):
		if id in [id1, id2]:
			plt.plot(*zip(*segment), color = 'green', marker='*', linewidth = 3)
		else:
			plt.plot(*zip(*segment), color = 'teal', marker='x', alpha = 0.7)    
		

	plt.show()

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

def timeAnalysis():
	box_size = 10000
	random.seed(1)
	numturns = 20
	meantimes = []
	numsegrange = range(100,5000,100)
	length = box_size/2

	for numseg in numsegrange:
		print("number of segments:", numseg)
		meantime = 0
		cell_size = box_size//math.floor(math.sqrt(numseg))
		#length = box_size/numseg/10
		for i in range(numturns):
			start_time = time.time()
			#segments = [generateRandomSegment(box_size,max_length) for i in range(num_segments)]
			#segments = [Segment.from_list(generateRandomSegmentBox(i,i+cell_size,j,j+cell_size)) for i in range(0,box_size,cell_size) for j in range(0,box_size,cell_size)]
			segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length)) for i in range(numseg)]

			#detected = checkSegmentsIntersectionTrivial(segments)
			detected, id1, id2 = detectIntersection(segments)
			meantime += time.time() - start_time

		meantime /= numturns
		meantimes.append(meantime)
	
	#plotdata = [(box_size/cell_size)**2 for cell_size in cellsizes]
	xaxis = numsegrange

	with open('plotdata/detection_verylarge.txt', 'w') as f:
		for line in list(zip(xaxis, meantimes)):
			f.write(f"{line}\n")

	plt.plot(xaxis, meantimes)
	plt.show()

def checkIntersectionMain():
	num_segments = 100000
	box_size = 1000
	length = box_size*0.001
	varlength = box_size*0.001
	cell_size = 5
	#random.seed(1)
	#segments = [generateRandomSegment(box_size,max_length) for i in range(num_segments)]
	segments = [generateRandomSegmentBox(i,i+cell_size,j,j+cell_size) for i in range(0,box_size,cell_size) for j in range(0,box_size,cell_size)]
	#segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length,varlength)) for i in range(num_segments)]

	start_time = time.time()
	#print(checkSegmentsIntersectionTrivial(segments))
	detected, id1, id2 = detectIntersection(segments)
	elapsed = time.time() - start_time
	print("Time:", elapsed)
	print("Detected =", detected)

def testeMateus():
	segmentos = [[(30, 62), (53, 49)],
             [(63, 35), (85, 63)],
             [(25, 52), (31, 22)],
             [(20, 81), (66, 54)],
             [(58, 75), (28, 32)]]

	root_sweep = None

	Sweep_Line = AVL_Tree()

	root_sweep = Sweep_Line.insert(root_sweep, 2, 5)
	root_sweep = Sweep_Line.insert(root_sweep, 4, 7)
	root_sweep = Sweep_Line.insert(root_sweep, 3, 8)

	print(Sweep_Line.inOrder(root_sweep))

	root_sweep = Sweep_Line.delete(root_sweep,4,7)

	print(Sweep_Line.inOrder(root_sweep))

if __name__ == "__main__":
	#plotsegmentsMain()
	#testSegmentClass()
	#checkIntersectionMain()
	#timeAnalysis()
	testeMateus()