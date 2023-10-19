from enum import Enum
from functools import cmp_to_key
import heapq
from math import cos, pi, sin
import random

from matplotlib import pyplot as plt
import numpy as np
from orientation_utils import compare_ccw

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
		return 0.5*(self.p1 + self.p2)
	
	def evalX(self, x):
		A = self.p2[1] - self.p1[1]
		B = self.p1[0] - self.p2[0]
		C = A*self.p1[0] + B*self.p1[1]
		return (C-A*x)/B

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

class Node(object):
	def __init__(self, id, comparator):
		self.val = id
		self.left = None
		self.right = None
		self.height = 1
		self.parent = None
		self.comparator = comparator

	def __lt__(self, other):
		return self.comparator(self.val, other.val)

class StateTree(object):
 
	def __init__(self, segments : list[Segment], comparator):
		self.segments = segments
		self.comparator = comparator

	def insert(self, root, key):
		
		# Step 1 - Perform normal BST
		node = Node(key, self.comparator)
		if not root:
			return Node(key, self.comparator)
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

		node = Node(key, self.comparator)
		# Step 1 - Perform standard BST delete
		if not root:
			return root

		elif node < root:
			root.left = self.delete(root.left, key)

		elif node > root:
			root.right = self.delete(root.right, key)

		else:
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
		node = Node(key, self.comparator)

		if not root:
			return None
		
		elif node < root:
			return self.getNodeById(root.left, key)

		elif node > root:
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
		self.currentEvent = self.eventQueue[0]
		self.comparator = lambda i,j : self.segments[i].evalX(self.currentEvent) < self.segments[j].evalX(self.currentEvent)
		self.stateTree = StateTree(self.segments, self.comparator)
		self.root = None
	
	def identify(self):
		intersections = []

		while self.eventQueue:
			xvalue, yvalue, event = heapq.heappop(self.eventQueue)
			self.currentEvent = xvalue
			print("In xvalue", xvalue, "and event type", event.type.value)
			self.stateTree.seeTree(self.root)
			if event.type == EventType.ADD:
				i = event.id[0]
				#yvalue = self.segments[i][0][1]
				self.root = self.stateTree.insert(self.root, i)
				curNode = self.stateTree.getNodeById(self.root,i)
				nextNode = self.stateTree.getNextNode(curNode)
				prevNode = self.stateTree.getPreviousNode(curNode)
				id1 = prevNode.val if prevNode is not None else None
				id2 = nextNode.val if nextNode is not None else None
				
				if id1 is not None and checkSegmentIntersection(self.segments[id1], self.segments[i]) == True:
					idtuple = (id1,i) if id1 < i else (i,id1)
					intersections.append(idtuple)
					xintersection,yintersection = getXYIntersection(self.segments[id1], self.segments[i])
					heapq.heappush(self.eventQueue, (xintersection,yintersection, Event(EventType.INTERSECTION, idtuple)))
				if id2 is not None and checkSegmentIntersection(self.segments[id2], self.segments[i]) == True:
					idtuple = (id2, i) if id2 < i else (i, id2)
					intersections.append(idtuple)
					xintersection,yintersection = getXYIntersection(self.segments[id2], self.segments[i])
					heapq.heappush(self.eventQueue, (xintersection,yintersection, Event(EventType.INTERSECTION, idtuple)))

			elif event.type == EventType.DELETE:
				i = event.id[0]
				#yvalue = self.segments[i][1][1]
				curNode = self.stateTree.getNodeById(self.root,i)
				nextNode = self.stateTree.getNextNode(curNode)
				prevNode = self.stateTree.getPreviousNode(curNode)
				id1 = prevNode.val if prevNode is not None else None
				id2 = nextNode.val if nextNode is not None else None
				
				if id1 is None or id2 is None:
					continue

				if checkSegmentIntersection(self.segments[id1], self.segments[id2]) == True:
					idtuple = (id2, id1) if id2 < id1 else (id1, id2)
					intersections.append(idtuple)
					xintersection,yintersection = getXYIntersection(self.segments[id2], self.segments[id1])
					heapq.heappush(self.eventQueue, (xintersection,yintersection, Event(EventType.INTERSECTION, idtuple)))

				self.root = self.stateTree.delete(self.root, i)

			elif event.type == EventType.INTERSECTION:
				id1,id2 = event.id
				print("Intersection",id1,id2)


			else:
				print("Event type is not valid :", event.type)

		return intersections




if __name__ == "__main__":
	s1 = [
		[(1,10),(9,1)],[(2,13),(8,7)],[(5,8),(10,8)]
	]
	seglist = [
		[(1,7),(8,5)],[(2,5),(7,3)],[(3,1),(9,12)],[(6.5,6),(10,7)],[(1,1),(4.1,3.2)] 
	]
	segments = [Segment.from_list(seg) for seg in s1]

	print(getXYIntersection(segments[1], segments[2]))

	box_size = 100
	length = box_size*0.1
	varlength = box_size*0.01
	num_segments = 100
	segments = [Segment.from_list(generateRandomSegmentRangeLen(box_size,length,varlength)) for i in range(num_segments)]
	
	random.shuffle(segments)

	bentley = BentleyOttman(segments)

	ids = bentley.identify()
	print(ids)

	for id,segment in enumerate(segments):
		plt.text(*segment.getCentroid(), str(id))
		if id in [it for subl in ids for it in subl]:
			plt.plot(*zip(*segment), color = 'red', marker='*', linewidth = 3)
		else:
			plt.plot(*zip(*segment), color = 'teal', marker='x', alpha = 0.7)    
	plt.show()