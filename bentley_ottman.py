from enum import Enum
from functools import cmp_to_key
import heapq
from math import cos, pi, sin
import math
import random
import time

from matplotlib import pyplot as plt
import numpy as np
from orientation_utils import compare_ccw
from utils import openAndTreatFile

def checkSegmentIntersection(segment1, segment2):
	ccw1 = compare_ccw(segment1[0], segment1[1], segment2[0])*compare_ccw(segment1[0], segment1[1], segment2[1])
	ccw2 = compare_ccw(segment2[0], segment2[1], segment1[0])*compare_ccw(segment2[0], segment2[1], segment1[1])

	if ccw1 == 0 or ccw2 == 0:
		return ccw1 < 0 or ccw2 < 0
	return ccw1 < 0 and ccw2 < 0

def getXYIntersection(segment1, segment2):
	(x1,y1),(x2,y2) = segment1
	(x3,y3),(x4,y4) = segment2

	A, B = y1 - y2, x2 - x1
	C = A*x2 + B*y2
	D, E = y3 - y4, x4 - x3
	F = D*x4 + E*y4

	#print(A, B, C, D, E, F)

	det = (B*D - A*E)
	xintersection = (B*F - C*E)/det
	yintersection = (C*D - A*F)/det

	#print(det)
	#print(xintersection, yintersection)

	return xintersection, yintersection

def getxyintersection(segment1, segment2):
	pass

class Point(tuple):
	def __add__(self, other):
		return Point(a+b for a,b in zip(self, other))
	def __sub__(self, other):
		return Point(a-b for a,b in zip(self, other))
	def __rmul__(self,other):
		return Point(other*a for a in self)
	def __lt__(self, other):
		return self[0] < other[0]

class Segment:
	def __init__(self, point1 : Point, point2 : Point):
		self.p1 = point1 if point1 <= point2 else point2
		self.p2 = point2 if self.p1 == point1 else point1

	@classmethod
	def from_list(cls, segmentList):
		return cls(*[Point(tup) for tup in segmentList])

	def __getitem__(self,key):
		if key in (0,1):
			return self.p1 if key == 0 else self.p2
		raise IndexError

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
		return 0.5*(self.p1 + self.p2)

	def evalX(self, x):
		A = self.p2[1] - self.p1[1]
		B = self.p1[0] - self.p2[0]
		C = A*self.p1[0] + B*self.p1[1]
		if B == 0:
			return math.inf
		val = (C-A*x)/B
		return val

# def setSegmentByList(seglist : list[Segment]):
# 	return Segment(seglist[0],seglist[1])

# def getEventList(segments : list[Segment]):
# 	L = len(segments)
# 	sortedList = [i for i in range(2*L)]
# 	cmpLambda = lambda i,j : -1 if segments[i%L][i//L][0] < segments[j%L][j//L][0] else (1 if segments[i%L][i//L][0] > segments[j%L][j//L][0] else 0)
# 	sortedList.sort(key=cmp_to_key(cmpLambda))

# 	return sortedList

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

class EventType(Enum):
	ADD = 1
	DELETE = 2
	INTERSECTION = 3
	def __lt__(self, other):
		if self.__class__ is other.__class__:
			return self.value < other.value
		return NotImplemented

class Event:
	def __init__(self, type : EventType, segmentids : tuple):
		self.type = type
		self.id = segmentids
	def __lt__(self, other):
		return self.type < other.type

def getEventQueue(segments : list[Segment]):
	heap = []
	for i in range(len(segments)):
		heapq.heappush(heap, (segments[i][0][0],segments[i][0][1], Event(EventType.ADD, (i,))))
		heapq.heappush(heap, (segments[i][1][0],segments[i][1][1], Event(EventType.DELETE, (i,))))
	return heap

# Suboptimal data structure for State
class StateList:
	def __init__(self, lst):
		self._lst = lst
		self.len = len(lst)
	def __getitem__(self, item):
		return self._lst[item]
	def __setitem__(self, key, value):
		self._lst[key] = value
	def add(self, item):
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
	def swap(self,i,j):
		self._lst[i], self._lst[j] = self._lst[j], self._lst[i] 

class Node(object):
	def __init__(self, id, lt, gt):
		self.val = id
		self.left = None
		self.right = None
		self.height = 1
		self.parent = None
		self.lt = lt
		self.gt = gt

	def __lt__(self, other):
		return self.lt(self.val, other.val)
	def __gt__(self, other):
		return self.gt(self.val, other.val)

class StateTree(object):

	def __init__(self, segments : list[Segment], lt, gt):
		self.segments = segments
		self.lt = lt
		self.gt = gt
		self.size = 0

	def insert(self, root, key):
		# Step 1 - Perform normal BST
		node = Node(key, self.lt, self.gt)
		if not root:
			self.size += 1
			return Node(key, self.lt, self.gt)
		elif node < root:
			root.left = self.insert(root.left, key)
			root.left.parent = root
		else:
			root.right = self.insert(root.right, key)
			root.right.parent = root

		# Step 2 - Update the height of the
		# ancestor node
		root.height = 1 + max(self.getHeight(root.left),
						self.getHeight(root.right))

		# Step 3 - Get the balance factor
		balance = self.getBalance(root)

		# Step 4 - If the node is unbalanced,
		# then try out the 4 cases
		# Case 1 - Left Left
		if balance > 1 and node < root.left:
			return self.rightRotate(root)

		# Case 2 - Right Right
		if balance < -1 and node > root.right:
			return self.leftRotate(root)

		# Case 3 - Left Right
		if balance > 1 and node > root.left:
			root.left = self.leftRotate(root.left)
			return self.rightRotate(root)

		# Case 4 - Right Left
		if balance < -1 and node < root.right:
			root.right = self.rightRotate(root.right)
			return self.leftRotate(root)

		return root

	# Recursive function to delete a node with
	# given key from subtree with given root.
	# It returns root of the modified subtree.
	def delete(self, root, key):
		node = Node(key, self.lt, self.gt)
		# Step 1 - Perform standard BST delete
		if not root:
			return root

		elif node < root:
			root.left = self.delete(root.left, key)

		elif node > root:
			root.right = self.delete(root.right, key)

		else:
			self.size -= 1
			if root.left is None:
				temp = root.right
				if temp:
					temp.parent = root.parent if root else None
				root = None
				return temp

			elif root.right is None:
				temp = root.left
				if temp:
					temp.parent = root.parent if root else None
				root = None
				return temp

			temp = self.getMinValueNode(root.right)
			root.val = temp.val
			root.right = self.delete(root.right, temp.val)

		# If the tree has only one node,
		# simply return it
		if root is None:
			return root

		# Step 2 - Update the height of the
		# ancestor node
		root.height = 1 + max(self.getHeight(root.left),
							self.getHeight(root.right))

		# Step 3 - Get the balance factor
		balance = self.getBalance(root)

		# Step 4 - If the node is unbalanced,
		# then try out the 4 cases
		# Case 1 - Left Left
		if balance > 1 and self.getBalance(root.left) >= 0:
			return self.rightRotate(root)

		# Case 2 - Right Right
		if balance < -1 and self.getBalance(root.right) <= 0:
			return self.leftRotate(root)

		# Case 3 - Left Right
		if balance > 1 and self.getBalance(root.left) < 0:
			root.left = self.leftRotate(root.left)
			return self.rightRotate(root)

		# Case 4 - Right Left
		if balance < -1 and self.getBalance(root.right) > 0:
			root.right = self.rightRotate(root.right)
			return self.leftRotate(root)

		return root

	def leftRotate(self, z):

		y = z.right
		T2 = y.left

		# Perform rotation
		y.left = z
		z.right = T2

		# Update parents
		y.parent = z.parent
		z.parent = y
		if T2 is not None:
			T2.parent = z

		# Update heights
		z.height = 1 + max(self.getHeight(z.left),
						self.getHeight(z.right))
		y.height = 1 + max(self.getHeight(y.left),
						self.getHeight(y.right))

		# Return the new root
		return y

	def rightRotate(self, z):

		y = z.left
		T3 = y.right

		# Perform rotation
		y.right = z
		z.left = T3

		# Update parents
		y.parent = z.parent
		z.parent = y
		if T3 is not None:
			T3.parent = z

		# Update heights
		z.height = 1 + max(self.getHeight(z.left),
						self.getHeight(z.right))
		y.height = 1 + max(self.getHeight(y.left),
						self.getHeight(y.right))

		# Return the new root
		return y

	def getHeight(self, root):
		if not root:
			return 0

		return root.height

	def getBalance(self, root):
		if not root:
			return 0

		return self.getHeight(root.left) - self.getHeight(root.right)

	def getMinValueNode(self, root):
		if root is None or root.left is None:
			return root

		return self.getMinValueNode(root.left)

	def getMaxValueNode(self, root):
		if root is None or root.right is None:
			return root

		return self.getMaxValueNode(root.right)

	def getNodeById(self, root, key):
		node = Node(key, self.lt, self.gt)

		if not root:
			return None

		elif root > node:
			return self.getNodeById(root.left, key)

		elif root < node:
			return self.getNodeById(root.right, key)

		else:
			return root

	def getPreviousNode(self, root):
		if root is None:
			return None
		if root.left is not None:
			return self.getMaxValueNode(root.left)
		auxNode = root
		while auxNode.parent is not None:
			if auxNode.parent.right == auxNode:
				return auxNode.parent
			auxNode = auxNode.parent
		return None

	def getNextNode(self, root):
		if root is None:
			return None
		if root.right is not None:
			return self.getMinValueNode(root.right)
		auxNode = root
		while auxNode.parent is not None:
			if auxNode.parent.left == auxNode:
				return auxNode.parent
			auxNode = auxNode.parent
		return None

	def swapNodes(self, node1, node2):
		if node1 is None or node2 is None:
			#print("Node is none in swap method")
			return
		node1.val, node2.val = node2.val, node1.val

	def preOrder(self, root):

		if not root:
			return

		print("({0},height={1}) ".format(root.val, root.height), end="")
		self.preOrder(root.left)
		self.preOrder(root.right)

	def inOrder(self, root):
		if root:
			self.inOrder(root.left)
			print("ID: {}".format(root.val))
			self.inOrder(root.right)

	def seeTree(self, root):
		if not root:
			return
		print("Id: {} - Left Id: {}, Right Id: {} - Parent Id: {}".format(root.val,
				(root.left.val) if root.left else None, (root.right.val) if root.right else None,
				(root.parent.val) if root.parent else None))
		self.seeTree(root.left)
		self.seeTree(root.right)


class BentleyOttman:
	def __init__(self, segments : list[Segment]):
		self.segments = segments
		self.eventQueue = getEventQueue(self.segments)
		self.currentStep = 0
		self.currentEvent = self.eventQueue[0]
		self.currentNonIntersectionEvent = None
		self.previousEvent = None
		self.lt = lambda i,j : self.segments[i].evalX(self.currentEvent) < self.segments[j].evalX(self.currentEvent)
		self.gt = lambda i,j : self.segments[i].evalX(self.currentEvent) > self.segments[j].evalX(self.currentEvent)
		self.stateTree = StateTree(self.segments, self.lt, self.gt)
		#self.stateList = StateList([])
		self.root = None
		self.stateTreeSize = 0

	def identify(self):
		intersections = {}
		statesizeovertime = []
		heightovertime = []

		while self.eventQueue:
			xvalue, yvalue, event = heapq.heappop(self.eventQueue)
			self.currentEvent = xvalue
			#print("In xvalue", xvalue, "and event type", event.type.value)
			#self.stateTree.seeTree(self.root)
			if event.type == EventType.ADD:
				self.currentStep += 1
				self.currentNonIntersectionEvent = xvalue
				self.stateTreeSize += 1
				i = event.id[0]
				#yvalue = self.segments[i][0][1]
				self.root = self.stateTree.insert(self.root, i)
				statesizeovertime.append((self.currentStep,self.stateTreeSize))
				heightovertime.append((self.currentStep, self.stateTree.getHeight(self.root)))
				curNode = self.stateTree.getNodeById(self.root,i)
				nextNode = self.stateTree.getNextNode(curNode)
				prevNode = self.stateTree.getPreviousNode(curNode)
				id1 = prevNode.val if prevNode is not None else None
				id2 = nextNode.val if nextNode is not None else None

				# DELETE THE LINE BELOW
				if id1 == i or id2 == i:
					continue

				if id1 is not None and checkSegmentIntersection(self.segments[id1], self.segments[i]) == True:
					idtuple = (id1,i) if id1 < i else (i,id1)
					if idtuple in intersections:
						continue
					#print("intersection", idtuple)
					xintersection,yintersection = getXYIntersection(self.segments[id1], self.segments[i])
					if idtuple not in intersections:
						intersections[idtuple] = (xintersection,yintersection)
					if True:#xintersection > self.currentEvent:
						heapq.heappush(self.eventQueue, (xintersection,yintersection, Event(EventType.INTERSECTION, idtuple)))
				if id2 is not None and checkSegmentIntersection(self.segments[id2], self.segments[i]) == True:
					idtuple = (id2, i) if id2 < i else (i, id2)
					if idtuple in intersections:
						continue
					#print("intersection", idtuple)
					xintersection,yintersection = getXYIntersection(self.segments[id2], self.segments[i])
					if xintersection > self.currentEvent and idtuple not in intersections:
						intersections[idtuple]=(xintersection,yintersection)
						heapq.heappush(self.eventQueue, (xintersection,yintersection, Event(EventType.INTERSECTION, idtuple)))

			elif event.type == EventType.DELETE:
				self.currentStep += 1
				self.currentNonIntersectionEvent = xvalue
				i = event.id[0]
				#yvalue = self.segments[i][1][1]
				curNode = self.stateTree.getNodeById(self.root,i)
				nextNode = self.stateTree.getNextNode(curNode)
				prevNode = self.stateTree.getPreviousNode(curNode)
				id1 = prevNode.val if prevNode is not None else None
				id2 = nextNode.val if nextNode is not None else None

				if id1 is None or id2 is None:
					continue

				# DELETE THE LINE BELOW
				if id1 == id2:
					continue

				if checkSegmentIntersection(self.segments[id1], self.segments[id2]) == True:
					idtuple = (id2, id1) if id2 < id1 else (id1, id2)
					if idtuple in intersections:
						continue
					#print("intersection", idtuple)
					xintersection,yintersection = getXYIntersection(self.segments[id2], self.segments[id1])
					if idtuple not in intersections:
						intersections[idtuple] = (xintersection,yintersection)
					if True:#xintersection > self.currentEvent:
						heapq.heappush(self.eventQueue, (xintersection,yintersection, Event(EventType.INTERSECTION, idtuple)))

				self.root = self.stateTree.delete(self.root, i)
				self.stateTreeSize -= 1
				statesizeovertime.append((self.currentStep,self.stateTreeSize))
				heightovertime.append((self.currentStep, self.stateTree.getHeight(self.root)))

			elif event.type == EventType.INTERSECTION:
				i,j = event.id
				tmpcurevent = self.currentEvent
				self.currentEvent = self.previousEvent
				node1 = self.stateTree.getNodeById(self.root, i)
				node2 = self.stateTree.getNodeById(self.root, j)
				self.stateTree.swapNodes(node1, node2)

				if node1 is None or node2 is None:
					continue

				n1 = node1 if node1 < node2 else node2 # it is above, using pre swap comparison
				n2 = node1 if n1 == node2 else node2

				nextNode = self.stateTree.getNextNode(n1)
				prevNode = self.stateTree.getPreviousNode(n2)
				id1 = prevNode.val if prevNode is not None else None
				id2 = nextNode.val if nextNode is not None else None

				self.currentEvent = tmpcurevent

				# DELETE THE LINE BELOW
				if id1 == i or id1 == j or id2 == i or id2 == j:
					continue

				if id1 is not None and id1 != i and checkSegmentIntersection(self.segments[id1], self.segments[i]) == True:
					idtuple = (i, id1) if i < id1 else (id1, i)
					if idtuple in intersections:
						continue
					#print("intersection", idtuple)
					xintersection,yintersection = getXYIntersection(self.segments[i], self.segments[id1])
					if idtuple not in intersections:
						intersections[idtuple] = (xintersection,yintersection)
					if True:#xintersection > self.currentEvent:
						heapq.heappush(self.eventQueue, (xintersection,yintersection, Event(EventType.INTERSECTION, idtuple)))
				if id1 is not None and id1 != j and checkSegmentIntersection(self.segments[id1], self.segments[j]) == True:
					idtuple = (j, id1) if j < id1 else (id1, j)
					if idtuple in intersections:
						continue
					#print("intersection", idtuple)
					xintersection,yintersection = getXYIntersection(self.segments[j], self.segments[id1])
					if idtuple not in intersections:
						intersections[idtuple] = (xintersection,yintersection)
					if True:#xintersection > self.currentEvent:
						heapq.heappush(self.eventQueue, (xintersection,yintersection, Event(EventType.INTERSECTION, idtuple)))
				if id2 is not None and id2 != i and checkSegmentIntersection(self.segments[id2], self.segments[i]) == True:
					idtuple = (i, id2) if i < id2 else (id2, i)
					if idtuple in intersections:
						continue
					#print("intersection", idtuple)
					xintersection,yintersection = getXYIntersection(self.segments[i], self.segments[id2])
					if idtuple not in intersections:
						intersections[idtuple] = (xintersection,yintersection)
					if True:#xintersection > self.currentEvent:
						heapq.heappush(self.eventQueue, (xintersection,yintersection, Event(EventType.INTERSECTION, idtuple)))
				if id2 is not None and id2 != j and checkSegmentIntersection(self.segments[id2], self.segments[j]) == True:
					idtuple = (j, id2) if j < id2 else (id2, j)
					if idtuple in intersections:
						continue
					#print("intersection", idtuple)
					xintersection,yintersection = getXYIntersection(self.segments[j], self.segments[id2])
					if idtuple not in intersections:
						intersections[idtuple] = (xintersection,yintersection)
					if True:#xintersection > self.currentEvent:
						heapq.heappush(self.eventQueue, (xintersection,yintersection, Event(EventType.INTERSECTION, idtuple)))

				#print("Intersection",id1,id2)


			else:
				print("Event type is not valid :", event.type)

			self.previousEvent = self.currentEvent

		return intersections, statesizeovertime, heightovertime

def identifyIntersectionTrivial(segments):
	ids = {}
	for i in range(len(segments)):
		for j in range(i+1, len(segments)):
			if checkSegmentIntersection(segments[i], segments[j]):
				ids[(i,j)] = getXYIntersection(segments[i], segments[j])
	return ids

def testMain():
	s1 = [
		[(1,10),(9,1)],[(2,13),(8,7)],[(5,8),(10,8)]
	]
	seglist = [
		[(1,7),(8,5)],[(2,5),(7,3)],[(3,1),(9,12)],[(6.5,6),(10,7)],[(1,1),(4.1,3.2)]
	]
	# s2 = [
	# 	[(1,1+j),(9,1+j)] for j in range(20)
	# ]+[
	# 	[(1.1 + x,0.9),(1.1 + x,9)] for x in range(20)
	# ]
	segments = [Segment.from_list(seg) for seg in seglist]

	#print(getXYIntersection(segments[1], segments[2]))

	random.seed(0)
	box_size = 100
	length = box_size*0.01
	varlength = box_size*0.01
	num_segments = 10000
	segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length,varlength)) for i in range(num_segments)]

	#random.shuffle(segments)

	# for id,segment in enumerate(segments):
	# 	#plt.text(*segment.getCentroid(), str(id))
	# 	plt.plot(*zip(*segment), color = 'teal', marker='s')
	# plt.show()

	# start = time.perf_counter()
	# idstrivial = identifyIntersectionTrivial(segments)
	# elapsed = 1000*(time.perf_counter() - start)
	# print("elapsed trivial = ", elapsed, "ms")

	# for id,segment in enumerate(segments):
	# 	#plt.text(*segment.getCentroid(), str(id))
	# 	if id in [it for subl in idstrivial for it in subl]:
	# 		plt.plot(*zip(*segment), color = 'red', marker='*', linewidth = 2, alpha = 0.7)
	# 		#plt.scatter(*zip(idstrivial[key]), color="fuchsia", marker="x", s=30)
	# 	else:
	# 		plt.plot(*zip(*segment), color = 'teal', marker='x', alpha = 0.7)
	# plt.show()

	start = time.perf_counter()
	bentley = BentleyOttman(segments)

	ids, statesizes, heights = bentley.identify()
	elapsed = 1000*(time.perf_counter() - start)
	#ids.sort(key=cmp_to_key(lambda i,j : 1 if i[0] > j[0] else (-1 if i[0] < j[0] else 0)))
	print(ids)
	print("Elapsed =", elapsed, "ms")

	for id,segment in enumerate(segments):
		#plt.text(*segment.getCentroid(), str(id))
		if id in [it for subl in ids for it in subl]:
			for key, (x,y) in ids.items():
				if id in key:
					plt.plot(*zip(*segment), color = 'red', marker='*', linewidth = 2, alpha = 0.7)
					plt.scatter(*zip(ids[key]), color="fuchsia", marker="x", s=30)
		else:
			plt.plot(*zip(*segment), color = 'teal', marker='x', alpha = 0.7)
	plt.show()
	
	# plt.plot(*zip(*statesizes), color="red")
	# plt.xlabel("Num steps")
	# plt.ylabel("State size")
	# plt.title("Number of active segments over time")
	# plt.show()

	# plt.plot(*zip(*heights), color="purple")
	# plt.xlabel("Num steps")
	# plt.ylabel("State tree height")
	# plt.title("State Tree Height over time")
	# plt.show()

def timeAnalysis():
	box_size = 1000
	random.seed(1)
	numturns = 5
	meantimes = []
	maxseg = 2000
	segstep = 100
	numsegrange = range(segstep,maxseg,segstep)
	length = box_size*0.01
	varlength = length*0.1
	for numseg in numsegrange:
		print("number of segments:", numseg)
		meantime = 0
		#length = box_size/numseg/10
		for i in range(numturns):
			segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length,varlength)) for _ in range(numseg)]
			#segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length)) for i in range(numseg)]
			start_time = time.perf_counter()
			#ids = identifyIntersectionTrivial(segments)
			bentley = BentleyOttman(segments)
			ids, statesizes, heights = bentley.identify()
			meantime += time.perf_counter() - start_time

		meantime /= numturns
		meantimes.append(meantime)
	
	#plotdata = [(box_size/cell_size)**2 for cell_size in cellsizes]
	xaxis = numsegrange

	#with open('plotdata/detection_trivial_length_'+str(length)+'.txt', 'w') as f:
	with open('plotdata/identification_avl_'+str(maxseg)+'-'+str(segstep)+'numturns'+str(numturns)+'length'+str(length)+'.txt', 'w') as f:
		for line in list(zip(xaxis, meantimes)):
			f.write(f"{line}\n")

	plt.plot(xaxis, meantimes)
	plt.show()

def logplots():
	line1 = openAndTreatFile("plotdata/identification_trivial_2000-100numturns5length10.0.txt")
	line2 = openAndTreatFile("plotdata/identification_avl_5000-100numturns3.txt")
	
	plt.loglog(*line1, color = 'blue')
	plt.loglog(*line2, color = 'green')
	#plt.loglog(*line3, color = 'red')
	
	slope1, intercept1 = np.polyfit(np.log(line1[0][1:]), np.log(line1[1][1:]), 1) 
	slope2, intercept2 = np.polyfit(np.log(line2[0][1:]), np.log(line2[1][1:]), 1) 
	#slope3, intercept3 = np.polyfit(np.log(line3[0][1:]), np.log(line3[1][1:]), 1)

	regline1 = [intercept1 + slope1*x for x in np.log(line1[0][1:])]
	regline2 = [intercept2 + slope2*x for x in np.log(line2[0][1:])]
	#regline3 = [intercept3 + slope3*x for x in np.log(line3[0][1:])]

	legend = ['Trivial', 'Sweep Line']
	plt.legend(legend)
	plt.xlabel("State size")
	plt.ylabel("Time (sec)")

	print("1 :", slope1, ": 2 :", slope2)#, ": 3 :", slope3)

	plt.show()

def timeAndSegSize():
	#num_segments = 1000
	box_size = 1000
	num_turns = 3
	seed = 2
	random.seed(seed)
	np.random.seed(seed)
	segsizerange = [0.1, 1, 10, 100, 1000]
	numsegrange = range(100,2000,100)

	for segsize in segsizerange:
		meantimes = []
		for numseg in numsegrange:
			print("numseg", numseg)
			meantime = 0
			for _ in range(num_turns):
				segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,segsize)) for i in range(numseg)]
				random.shuffle(segments)
				start_time = time.perf_counter()
				bentley = BentleyOttman(segments)
				ids, statesizes, heights = bentley.identify()
				meantime += time.perf_counter() - start_time
			meantime /= num_turns
			meantimes.append(meantime)

		xaxis = numsegrange
		with open('plotdata/identification_avl_seglen-'+str(segsize)+'numturns'+str(num_turns)+'.txt', 'w') as f:
			for line in list(zip(xaxis, meantimes)):
				f.write(f"{line}\n")

def stateandsegsize():
	numseg = 1000
	box_size = 1000
	num_turns = 1
	seed = 2
	random.seed(seed)
	np.random.seed(seed)
	segsizerange = [0.1, 1, 10, 100, 1000]

	legend = []

	for segsize in segsizerange:
		#meansizes = [0 for _ in range(2*numseg)]	
		segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,segsize)) for i in range(numseg)]
		random.shuffle(segments)
		bentley = BentleyOttman(segments)
		ids, statesizes, heights = bentley.identify()
		plt.plot(*zip(*heights))
		legend.append(str(segsize))
	
	plt.legend(legend)
	plt.show()	



def newlogplots():
	legend = []
	for seglen in [0.1, 1, 10, 100, 1000]:
		line = openAndTreatFile("plotdata/identification_avl_seglen-"+str(seglen)+"numturns3.txt")
		plt.plot(*line)
		legend.append("Segment size = "+str(seglen))
	plt.xlabel("Num segments")
	plt.ylabel("Time (sec)")
	plt.legend(legend)
	plt.show()


if __name__ == "__main__":
	#testMain()
	#timeAnalysis()
	#logplots()
	#timeAndSegSize()
	#newlogplots()
	stateandsegsize()