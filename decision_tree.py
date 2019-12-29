# Create a Decision Tree Class -- Every node in the Tree is an object of this class
class DecisionTree:

	def __init__(self):
		self.left = None
		self.right = None
		self.feature = None
		self.prediction = None
		self.is_leaf = None

	# Inserts a node into the tree
	def insert(self, feature, is_leaf, prediction):
		self.feature = feature
		self.is_leaf = is_leaf
		self.prediction = prediction