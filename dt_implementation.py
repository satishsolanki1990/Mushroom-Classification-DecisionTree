import numpy as np
from decision_tree import DecisionTree
from random import sample

def decision_tree(data, max_depth, depth, tree, m):

	if depth == max_depth:
		# If you have reached maximum depth, store the prediction based on number of positive and negative examples
		label = predictLabel(data[:, 0])
		tree.insert(None, True, label)
		return

	root_value = calculate_gini(data[:, 0])

	gain = 0
	best_feature = 0
	i = 0
	for feature_column in sample(range(1, data.shape[1]), m):

		current_gain = information_gain(data[:, 0], data[:, feature_column], root_value)
		if current_gain > gain:
			gain = current_gain
			best_feature = feature_column
		i = i + 1

	if gain == 0:
		label = predictLabel(data[:, 0])
		tree.insert(None, True, label)
		return

	feature_true = data[data[:, best_feature] == 1]
	feature_false = data[data[:, best_feature] == 0]

	label = predictLabel(data[:, 0])
	tree.insert(best_feature, False, label)

	depth = depth + 1

	tree.left = DecisionTree()
	tree.right = DecisionTree()

	decision_tree(feature_true, max_depth, depth, tree.left, m)
	decision_tree(feature_false, max_depth, depth, tree.right, m)


def predictLabel(data):
	sample_size = len(data)

	# The Number of unique values -- Count returns the number of unique counts
	# Get a list containing count of positive and negative labels
	unique, count = np.unique(data, return_counts = True)
	if len(unique) == 1:

		if unique[0] == 1:
			return 1
		else:
			return -1
	if unique[0] == 1:
		p_pos = count[0] / sample_size
		p_neg = count[1] / sample_size
	else:
		p_pos = count[1] / sample_size
		p_neg = count[0] / sample_size
	if p_pos >= p_neg:
		return 1
	else:
		return -1


def calculate_gini(data):
	sample_size = len(data)
	if sample_size == 0:
		return 0
	unique, count = np.unique(data, return_counts = True)
	if len(unique) == 1:
		return 0
	if unique[0] == 1:
		positive = count[0] / sample_size
		negative = count[1] / sample_size
	else:
		positive = count[1] / sample_size
		negative = count[0] / sample_size
 
	return (1 - (positive ** 2) - (negative ** 2))


def information_gain(label, data, u_root):
	value = np.empty([len(data), 2])
	value[:, 0] = label
	value[:, 1] = data

	gain = 0
	prev_label = 0

	val_t = value[value[:, 1] == 1]
	val_f = value[value[:, 1] == 0]

	feature_true = val_t
	feature_false = val_f

	u_left = calculate_gini(feature_true[:, 0])
	u_right = calculate_gini(feature_false[:, 0])

	p_left = len(feature_true) / len(value)
	p_right = len(feature_false) / len(value)

	gain = u_root - p_left * u_left - p_right * u_right

	return gain


def predict_result(tree, example, depth, max_depth):
	if tree.is_leaf:
		return tree.prediction
	if depth == max_depth:
		return tree.prediction

	# Recursively move down the tree to compute the prediction
	if example[tree.feature].reshape(1,1) == 1:
		return predict_result(tree.left, example, depth + 1, max_depth)
	else:
		return predict_result(tree.right, example, depth + 1, max_depth)


def accuracy_decision_tree(tree, data, depth):
	size = len(data)
	error = 0
	for row in range(0, size):
		predicted_value = predict_result(tree, data[row, :], 0, depth)
		if predicted_value != data[row, 0]:
			error = error + 1
	accuracy = ((size - error) / size) * 100
	return accuracy

def test_decision_tree(tree, data, depth):
	size = len(data)
	error = 0
	predicted_values = np.empty(0)
	for row in range(0, size):
		predicted_values = np.append(predicted_values,predict_result(tree, data[row, :], 0, depth))
	return predicted_values
