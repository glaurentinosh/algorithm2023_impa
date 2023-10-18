# Source code for AVL Tree: https://www.geeksforgeeks.org/deletion-in-an-avl-tree/
# Modified to fit my code

# Python code to delete a node in AVL tree
# Generic tree node class
class TreeNode(object):
	def __init__(self, id, val):
		self.id = id
		self.val = val
		self.left = None
		self.right = None
		self.height = 1
		self.parent = None

# AVL tree class which supports insertion,
# deletion operations
class AVL_Tree(object):

	def insert(self, root, id, key):
		
		# Step 1 - Perform normal BST
		if not root:
			return TreeNode(id, key)
		elif key < root.val:
			root.left = self.insert(root.left, id, key)
			root.left.parent = root
		else:
			root.right = self.insert(root.right, id, key)
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
		if balance > 1 and key < root.left.val:
			return self.rightRotate(root)

		# Case 2 - Right Right
		if balance < -1 and key > root.right.val:
			return self.leftRotate(root)

		# Case 3 - Left Right
		if balance > 1 and key > root.left.val:
			root.left = self.leftRotate(root.left)
			return self.rightRotate(root)

		# Case 4 - Right Left
		if balance < -1 and key < root.right.val:
			root.right = self.rightRotate(root.right)
			return self.leftRotate(root)

		return root

	# Recursive function to delete a node with
	# given key from subtree with given root.
	# It returns root of the modified subtree.
	def delete(self, root, id, key):

		# Step 1 - Perform standard BST delete
		if not root:
			return root

		elif key < root.val:
			root.left = self.delete(root.left, id, key)

		elif key > root.val:
			root.right = self.delete(root.right, id, key)

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
			root.id = temp.id
			root.right = self.delete(root.right, temp.id,
									temp.val)

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
	
	def getNodeById(self, root, id, key):
		if not root:
			return None

		elif key < root.val:
			return self.getNodeById(root.left, id, key)

		elif key > root.val:
			return self.getNodeById(root.right, id, key)

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
		node1.id, node2.id = node2.id, node1.id

	def preOrder(self, root):

		if not root:
			return

		print("({0},{1},height={2}) ".format(root.id, root.val, root.height), end="")
		self.preOrder(root.left)
		self.preOrder(root.right)

	def inOrder(self, root):
		if root:
			self.inOrder(root.left)
			print("ID: {}, Value: {}".format(root.id, root.val))
			self.inOrder(root.right)
		
	def seeTree(self, root):
		if not root:
			return
		print("Id, value: {}, {} - Left Id: {}, Right Id: {} - Parent Id: {}".format(root.id, root.val,
				(root.left.id, root.left.val) if root.left else None, (root.right.id, root.right.val) if root.right else None,
				(root.parent.id,root.parent.val) if root.parent else None))
		self.seeTree(root.left)
		self.seeTree(root.right)


if __name__ == "__main__":
	myTree = AVL_Tree()
	root = None
	print("Height", myTree.getHeight(root))

	# nums = [9, 5, 10, 0, 6, 11, -1, 1, 2]

	# for id, num in enumerate(nums):
	# 	root = myTree.insert(root, id, num)
	root = myTree.insert(root, 0, 3) 
	root = myTree.insert(root, 1, 9) 
	root = myTree.insert(root, 2, 0) 
	root = myTree.insert(root, 3, 11) 
	root = myTree.insert(root, 4, 12) 
	root = myTree.insert(root, 5, 1) 
	root = myTree.insert(root, 6, 4) 
	root = myTree.insert(root, 7, 6) 

	print("\nBefore deletion")
	myTree.seeTree(root)

	root = myTree.delete(root, 3, 11)
	print("\nAfter deletion")
	myTree.seeTree(root)

	# Delete
	# key = 10
	# root = myTree.delete(root, 2, key)

	# # Preorder Traversal
	# print("Preorder Traversal after deletion -")
	# myTree.preOrder(root)
	# print()

	# Smallest and largest
	print("Smallest by root")
	print(myTree.getMinValueNode(root).val)
	print(myTree.getMaxValueNode(root).val)
	print()

	curnode = myTree.getNodeById(root, 5,1)
	print("Next of 25", myTree.getNextNode(curnode).val)
	#print("Previous of 5", myTree.getPreviousNode(root, 1, 5).val)


	# This code is contributed by Ajitesh Pathak