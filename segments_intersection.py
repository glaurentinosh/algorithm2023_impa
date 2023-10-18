from functools import cmp_to_key
import math
from orientation_utils import compare_ccw
import random
from math import pi,cos,sin
import matplotlib.pyplot as plt
from trees import *
import numpy as np
import time
from enum import Enum
import heapq

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

class EventType(Enum):
	ADD = 1
	DELETE = 2
	INTERSECTION = 3

class Event:
	def __init__(self, type : EventType, segmentids : tuple):
		self.type = type
		self.id = segmentids

def getEventQueue(segments : list[Segment]):
	heap = []
	for i in range(len(segments)):
		heapq.heappush(heap, (segments[i][0][0], Event(EventType.ADD, (i,))))
		heapq.heappush(heap, (segments[i][1][0], Event(EventType.DELETE, (i,))))
	return heap

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

def identifyIntersectionPoor(segments : list[Segment]):
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

def identifyIntersection(segments : list[Segment]):
	eventheap = getEventQueue(segments)
	stateTree = AVL_Tree()
	root = None
	L = len(segments)
	intersections = []

	while eventheap:
		xvalue, event = heapq.heappop(eventheap)
		if event.type == EventType.ADD:
			i = event.id[0]
			yvalue = segments[i][0][1]
			root = stateTree.insert(root, i, yvalue)
			curNode = stateTree.getNodeById(root,i, yvalue)
			nextNode = stateTree.getNextNode(curNode)
			prevNode = stateTree.getPreviousNode(curNode)
			id1 = prevNode.id if prevNode is not None else None
			id2 = nextNode.id if nextNode is not None else None
			
			if id1 is not None and checkSegmentIntersection(segments[id1], segments[i]) == True:
				idtuple = (id1,i) if id1 < i else (i,id1)
				intersections.append(idtuple)
				yintersection = getYValueIntersection(segments[id1], segments[i])
				heapq.heappush(eventheap, (yintersection, Event(EventType.INTERSECTION, idtuple)))
			if id2 is not None and checkSegmentIntersection(segments[id2], segments[i]) == True:
				idtuple = (id2, i) if id2 < i else (i, id2)
				intersections.append(idtuple)
				yintersection = getYValueIntersection(segments[id2], segments[i])
				heapq.heappush(eventheap, (yintersection, Event(EventType.INTERSECTION, idtuple)))

		elif event.type == EventType.DELETE:
			i = event.id[0]
			yvalue = segments[i][1][1]
			curNode = stateTree.getNodeById(root,i,yvalue)
			nextNode = stateTree.getNextNode(curNode)
			prevNode = stateTree.getPreviousNode(curNode)
			id1 = prevNode.id if prevNode is not None else None
			id2 = nextNode.id if nextNode is not None else None
			
			if id1 is None or id2 is None:
				continue

			if checkSegmentIntersection(segments[id1], segments[id2]) == True:
				intersections.append((id1,id2) if id1 < id2 else (id2,id1))

			root = stateTree.delete(root, i, yvalue)

		elif event.type == EventType.INTERSECTION:
			id1,id2 = event.id


		else:
			print("Event type is not valid :", event.type)

	return intersections

	

def checkSegmentIntersection(segment1, segment2):
	ccw1 = compare_ccw(segment1[0], segment1[1], segment2[0])*compare_ccw(segment1[0], segment1[1], segment2[1])
	ccw2 = compare_ccw(segment2[0], segment2[1], segment1[0])*compare_ccw(segment2[0], segment2[1], segment1[1])

	if ccw1 == 0 or ccw2 == 0:
		return ccw1 < 0 or ccw2 < 0
	return ccw1 < 0 and ccw2 < 0

def getYValueIntersection(segment1, segment2):
	(x1,y1),(x2,y2) = segment1
	(x3,y3),(x4,y4) = segment2

	A, B = y1 - y2, x2 - x1
	C = A*x2 + B*y2
	D, E = y3 - y4, x4 - x3
	F = A*x4 + B*y4

	return (A*F - C*D)/(B*D - A*E)

def detectIntersectionTrivial(segments):
	for i in range(len(segments)):
		for j in range(i+1, len(segments)):
			if checkSegmentIntersection(segments[i], segments[j]):
				return True, i, j
	return False, None, None

def identifyIntersectionTrivial(segments):
	ids = []
	for i in range(len(segments)):
		for j in range(i+1, len(segments)):
			if checkSegmentIntersection(segments[i], segments[j]):
				ids.append((i,j))
	return ids

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
	num_segments = 10
	box_size = 1000
	length = box_size*0.2
	varlength = box_size*0.01
	cell_size = 1
	seed = 2
	timemult=1000
	statesize = 20
	ycell = box_size/statesize
	xcell = box_size*statesize/num_segments
	random.seed(seed)
	np.random.seed(seed)
	segments = [Segment.from_list(generateRandomSegment(box_size,length)) for i in range(num_segments)]
	#segments = [Segment.from_list(generateRandomSegmentBox(i,i+cell_size,j,j+cell_size)) for i in range(0,box_size,cell_size) for j in range(0,box_size,cell_size)]
	#segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length,varlength)) for i in range(num_segments)]
	#segments = [Segment.from_list(generateRandomSegmentBox(xcell*(i//statesize),xcell*(1+i//statesize),ycell*(i%statesize),(1+(i%statesize))*ycell)) for i in range(num_segments)]
	random.shuffle(segments)

	print("Length:", len(segments))
	print("Sweep Line AVL")
	start = time.perf_counter()
	ids = identifyIntersection(segments)
	elapsed = timemult*(time.perf_counter() - start)
	print(elapsed)
	# print("Sweep Line List")
	# start = time.perf_counter()
	# detected, id1, id2 = detectIntersectionList(segments)
	# elapsed = timemult*(time.perf_counter() - start)
	# print(detected)
	# print(elapsed)
	# print("Trivial")
	# start = time.perf_counter()
	# detected, Id1, Id2 = detectIntersectionTrivial(segments)
	# elapsed = timemult*(time.perf_counter() - start)
	# print(detected)
	# print(elapsed)

	for id,segment in enumerate(segments):
		if id in [elem for tup in ids for elem in tup]:
			plt.plot(*zip(*segment), color = 'red', marker='*', linewidth = 3)
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

def timeAndSegSize():
	#num_segments = 1000
	box_size = 1000
	num_turns = 3
	seed = 2
	random.seed(seed)
	np.random.seed(seed)
	segsizerange = [0.1, 1, 10, 100, 1000]
	numsegrange = range(100,1000,100)

	for segsize in segsizerange:
		meantimes = []
		for numseg in numsegrange:
			print("numseg", numseg)
			meantime = 0
			for _ in range(num_turns):
				segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,segsize)) for i in range(numseg)]
				random.shuffle(segments)
				start_time = time.perf_counter()
				detected, id1, id2 = detectIntersection(segments)
				meantime += time.perf_counter() - start_time
			meantime /= num_turns
			meantimes.append(meantime)

		xaxis = numsegrange
		with open('plotdata/detection_avl_false_seglen-'+str(segsize)+'numturns'+str(num_turns)+'.txt', 'w') as f:
			for line in list(zip(xaxis, meantimes)):
				f.write(f"{line}\n")


def timeAndStateSize():
	#num_segments = 1000
	box_size = 1000
	num_turns = 3
	seed = 2
	random.seed(seed)
	np.random.seed(seed)
	staterange = range(10,1000,100)
	numsegrange = range(1000,10000,1000)

	for statesize in staterange:
		meantimes = []
		for num_segments in numsegrange:
			print("state size:", statesize)
			meantime = 0
			ycell = box_size/statesize
			xcell = box_size*statesize/num_segments
			for _ in range(num_turns):
				segments = [Segment.from_list(generateRandomSegmentBox(xcell*(i//statesize),xcell*(1+i//statesize),ycell*(i%statesize),(1+(i%statesize))*ycell)) for i in range(num_segments)]
				random.shuffle(segments)
				start_time = time.perf_counter()
				detected, id1, id2 = detectIntersection(segments)
				meantime += time.perf_counter() - start_time
			meantime /= num_turns
			meantimes.append(meantime)

		xaxis = numsegrange

		#with open('plotdata/detection_trivial_length_'+str(length)+'.txt', 'w') as f:
		with open('plotdata/detection_avl_false_statesize-'+str(statesize)+'numturns'+str(num_turns)+'.txt', 'w') as f:
			for line in list(zip(xaxis, meantimes)):
				f.write(f"{line}\n")

	#plt.plot(xaxis, meantimes)
	#plt.show()

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

def newlogplots():
	legend = []
	for seglen in [0.1, 1, 10, 100, 1000]:
		line = openAndTreatFile("plotdata/detection_avl_false_seglen-"+str(seglen)+"numturns3.txt")
		plt.plot(*line)
		legend.append("Segment size = "+str(seglen))
	plt.xlabel("Num segments")
	plt.ylabel("Time (sec)")
	plt.legend(legend)
	plt.show()

def testExample():
	segments = [
		[(1,10),(9,1)],[(2,13),(8,7)],[(5,8),(10,8)]
	]
	
	random.shuffle(segments)

	detected,id1,id2 = detectIntersectionTrivial(segments)

	for id,segment in enumerate(segments):
		if id in [id1,id2]:
			plt.plot(*zip(*segment), color = 'red', marker='*', linewidth = 3)
		else:
			plt.plot(*zip(*segment), color = 'teal', marker='x', alpha = 0.7)    
	plt.show()



if __name__ == "__main__":
	testExample()
	#plotsegmentsMain()
	#testSegmentClass()
	#checkIntersectionMain()
	#timeAnalysis()
	#timeAndStateSize()
	#timeAndSegSize()
	#newlogplots()
	