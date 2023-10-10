from functools import cmp_to_key
import math
from orientation_utils import compare_ccw
import random
from math import pi,cos,sin
import matplotlib.pyplot as plt
from trees import *
import numpy as np
import time

from utils import openAndTreatFile

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

class StateList:
	def __init__(self, lst):
		self._lst = lst
	def __getitem__(self, item):
		return self._lst[item]
	def __setitem__(self, key, value):
		self._lst[key] = value
	def append(self, item):
		self._lst.append(item)
	def getNext(self, item):
		for i in range(len(self._lst)-1):
			if item == self._lst[i]:
				return self._lst[i+1]
		return None
	def getPrevious(self, item):
		for i in range(1,len(self._lst)):
			if item == self._lst[i]:
				return self._lst[i-1]
		return None
	def delete(self, item):
		self._lst.remove(item) 

def detectIntersectionList(segments : list[Segment]):
	eventList = getEventList(segments)
	state = StateList([])

	L = len(segments)

	for i in eventList:
		if i // L == 0:
			state.append((i, segments[i%L][i//L][1]))
			curItem = state[-1]
			nextItem = state.getNext(curItem)
			prevItem = state.getPrevious(curItem)
			id1 = prevItem[0] if prevItem is not None else None
			id2 = nextItem[0] if nextItem is not None else None
			
			if id1 is not None and checkSegmentIntersection(segments[id1%L], segments[i%L]) == True:
				return True, id1, i
			if id2 is not None and checkSegmentIntersection(segments[id2%L], segments[i%L]) == True:
				return True, id2, i
			
		elif i // L == 1:
			state.append((i, segments[i%L][i//L][1]))
			curItem = state[-1]
			nextItem = state.getNext(curItem)
			prevItem = state.getPrevious(curItem)
			id1 = prevItem[0] if prevItem is not None else None
			id2 = nextItem[0] if nextItem is not None else None
			
			if id1 is None or id2 is None:
				continue

			if checkSegmentIntersection(segments[id1%L], segments[id2%L]) == True:
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
			
			if id1 is not None and checkSegmentIntersection(segments[id1%L], segments[i%L]) == True:
				return True, id1, i
			if id2 is not None and checkSegmentIntersection(segments[id2%L], segments[i%L]) == True:
				return True, id2, i
			
		elif i // L == 1:
			curNode = stateTree.getNodeById(root,i,segments[i%L][i//L][1])
			nextNode = stateTree.getNextNode(curNode)
			prevNode = stateTree.getPreviousNode(curNode)
			id1 = prevNode.id if prevNode is not None else None
			id2 = nextNode.id if nextNode is not None else None
			
			if id1 is None or id2 is None:
				continue

			if checkSegmentIntersection(segments[id1%L], segments[id2%L]) == True:
				return True, id1, id2

			root = stateTree.delete(root, i, segments[i%L][i//L][1])

	return False, None, None

def checkSegmentIntersection(segment1, segment2):
	ccw1 = compare_ccw(segment1[0], segment1[1], segment2[0])*compare_ccw(segment1[0], segment1[1], segment2[1])
	ccw2 = compare_ccw(segment2[0], segment2[1], segment1[0])*compare_ccw(segment2[0], segment2[1], segment1[1])

	if ccw1 == 0 or ccw2 == 0:
		return ccw1 < 0 or ccw2 < 0
	return ccw1 < 0 and ccw2 < 0

def detectIntersectionTrivial(segments):
	for i in range(len(segments)):
		for j in range(i+1, len(segments)):
			if checkSegmentIntersection(segments[i], segments[j]):
				return True, i, j
	return False, None, None

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

	return segment

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
	num_segments = 2500
	box_size = 1000
	length = box_size*0.001
	varlength = box_size*0.001
	cell_size = 1
	seed = 2
	timemult=1000
	statesize = 20
	ycell = box_size/statesize
	xcell = box_size*statesize/num_segments
	random.seed(seed)
	np.random.seed(seed)
	#segments = [Segment.from_list(generateRandomSegment(box_size,length)) for i in range(num_segments)]
	#segments = [Segment.from_list(generateRandomSegmentBox(i,i+cell_size,j,j+cell_size)) for i in range(0,box_size,cell_size) for j in range(0,box_size,cell_size)]
	#segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length,varlength)) for i in range(num_segments)]
	segments = [Segment.from_list(generateRandomSegmentBox(xcell*(i//statesize),xcell*(1+i//statesize),ycell*(i%statesize),(1+(i%statesize))*ycell)) for i in range(num_segments)]
	random.shuffle(segments)

	print("Length:", len(segments))
	print("Sweep Line AVL")
	start = time.perf_counter()
	detected, id1, id2 = detectIntersection(segments)
	elapsed = timemult*(time.perf_counter() - start)
	print(detected)
	print(elapsed)
	print("Sweep Line List")
	start = time.perf_counter()
	detected, id1, id2 = detectIntersectionList(segments)
	elapsed = timemult*(time.perf_counter() - start)
	print(detected)
	print(elapsed)
	print("Trivial")
	start = time.perf_counter()
	detected, Id1, Id2 = detectIntersectionTrivial(segments)
	elapsed = timemult*(time.perf_counter() - start)
	print(detected)
	print(elapsed)

	for id,segment in enumerate(segments):
		plt.plot(*zip(*segment), color = 'teal', marker='x', alpha = 0.7)    
	if Id1 is not None and Id2 is not None:
		for id in [Id1, Id2]:
			segment = segments[id]
			plt.plot(*zip(*segment), color = 'red', marker='*', linewidth = 3)
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
	box_size = 1000
	random.seed(1)
	numturns = 2
	meantimes = []
	maxseg = 10000
	segstep = 100
	numsegrange = range(segstep,maxseg,segstep)
	length = box_size*0.01
	for numseg in numsegrange:
		print("number of segments:", numseg)
		meantime = 0
		cell_size = box_size//math.floor(math.sqrt(numseg))
		#length = box_size/numseg/10
		for i in range(numturns):
			segments = [Segment.from_list(generateRandomSegmentBox(i,i+cell_size,j,j+cell_size)) for i in range(0,box_size,cell_size) for j in range(0,box_size,cell_size)]
			#segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length)) for i in range(numseg)]
			start_time = time.perf_counter()
			detected, id1, id2 = detectIntersection(segments)
			meantime += time.perf_counter() - start_time

		meantime /= numturns
		meantimes.append(meantime)
	
	#plotdata = [(box_size/cell_size)**2 for cell_size in cellsizes]
	xaxis = numsegrange

	#with open('plotdata/detection_trivial_length_'+str(length)+'.txt', 'w') as f:
	with open('plotdata/detection_avl_false_'+str(maxseg)+'-'+str(segstep)+'numturns'+str(numturns)+'.txt', 'w') as f:
		for line in list(zip(xaxis, meantimes)):
			f.write(f"{line}\n")

	plt.plot(xaxis, meantimes)
	plt.show()

def timeAndStateSize():
	num_segments = 1000
	box_size = 1000
	num_turns = 3
	seed = 2
	random.seed(seed)
	np.random.seed(seed)
	staterange = range(1,1000,9)
	meantimes = []

	for statesize in staterange:
		print("state size:", statesize)
		meantime = 0
		ycell = box_size/statesize
		xcell = box_size*statesize/num_segments
		for _ in range(num_turns):
			segments = [Segment.from_list(generateRandomSegmentBox(xcell*(i//statesize),xcell*(1+i//statesize),ycell*(i%statesize),(1+(i%statesize))*ycell)) for i in range(num_segments)]
			random.shuffle(segments)
			start_time = time.perf_counter()
			detected, id1, id2 = detectIntersectionList(segments)
			meantime += time.perf_counter() - start_time
		meantime /= num_turns
		meantimes.append(meantime)

	xaxis = staterange

	#with open('plotdata/detection_trivial_length_'+str(length)+'.txt', 'w') as f:
	with open('plotdata/detection_list_false_statesizes-'+'numturns'+str(num_turns)+'.txt', 'w') as f:
		for line in list(zip(xaxis, meantimes)):
			f.write(f"{line}\n")

	plt.plot(xaxis, meantimes)
	plt.show()

def checkIntersectionMain():
	num_segments = 10000
	box_size = 1000
	length = box_size*0.001
	varlength = box_size*0.001
	cell_size = 100
	#random.seed(1)
	#segments = [Segment.from_list(generateRandomSegment(box_size,max_length)) for i in range(num_segments)]
	#segments = [Segment.from_list(generateRandomSegmentBox(i,i+cell_size,j,j+cell_size)) for i in range(0,box_size,cell_size) for j in range(0,box_size,cell_size)]
	segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length,varlength)) for i in range(num_segments)]

	start_time = time.time()
	#print(detectIntersectionTrivial(segments))
	detected, id1, id2 = detectIntersectionList(segments)
	elapsed = time.time() - start_time
	print("Time:", elapsed)
	print("Detected =", detected)

def logplots():
	line1 = openAndTreatFile("plotdata/detection_list_false_statesizes-numturns3.txt")
	#line1 = openAndTreatFile("plotdata/detection_avl_false_10000-100numturns2.txt")
	#line1 = openAndTreatFile("plotdata/detection_avl_false_5000-100.txt")
	line2 = openAndTreatFile("plotdata/detection_trivial_false_1000-100numturns2.txt")
	line3 = openAndTreatFile("plotdata/detection_list_false_2000-100numturns2.txt")
	
	plt.scatter(*line1, color = 'blue')
	#plt.loglog(*line2, color = 'green')
	#plt.loglog(*line3, color = 'red')
	
	slope1, intercept1 = np.polyfit(np.log(line1[0][1:]), np.log(line1[1][1:]), 1) 
	slope2, intercept2 = np.polyfit(np.log(line2[0][1:]), np.log(line2[1][1:]), 1) 
	slope3, intercept3 = np.polyfit(np.log(line3[0][1:]), np.log(line3[1][1:]), 1)

	regline1 = [intercept1 + slope1*x for x in np.log(line1[0][1:])]
	regline2 = [intercept2 + slope2*x for x in np.log(line2[0][1:])]
	regline3 = [intercept3 + slope3*x for x in np.log(line3[0][1:])]

	legend = ['Sweep Line', 'Trivial', 'Sweep Line']
	plt.legend(legend)
	plt.xlabel("State size")
	plt.ylabel("Time (sec)")

	print("1 :", slope1, ": 2 :", slope2, ": 3 :", slope3)

	plt.show()

if __name__ == "__main__":
	#plotsegmentsMain()
	#testSegmentClass()
	#checkIntersectionMain()
	#timeAnalysis()
	timeAndStateSize()
	logplots()
	