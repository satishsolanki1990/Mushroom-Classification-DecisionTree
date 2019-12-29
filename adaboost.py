import numpy as np
from dt_implementation import predict_result
from decision_tree import DecisionTree

# Create the adaboost tree
def adaboost(data, l, depth):
	size = len(data)

	# weight_vector contains the weights for each of the data elements
	weight_vector = np.zeros(size)
	weight_vector.fill(1.0/ size)

	# Insert weight_vector as column indexed at 1 in data
	data = np.insert(data, 1, weight_vector, axis = 1)

	tree_list = []
	alpha_list = []

	for weakLearner in range(l):
		tree = DecisionTree()

		create_adaboost(data, depth, 0, tree)
		error, weight_list = calculate_error(tree, data, depth)
		alpha = np.log( (1 - error) / error) / 2

		data[:, 1] = data[:, 1] * np.exp(alpha * np.array(weight_list))
		data[:, 1] = data[:, 1] / np.sum( data[:, 1] )

		tree_list.append(tree)
		alpha_list.append(alpha)

	return tree_list, alpha_list


def calculate_gini_index(data):
	size = len(data)
	if size == 0:
		return 0

	sort_data = data[np.argsort(data[:, 0])]
	positive_negative = np.split(sort_data, np.where(sort_data[:, 0] == 1)[0][:1])

	# If only one label is present in the dataset or if the total size if only 1; return 0 because no gain achieved
	if len(positive_negative[0]) == 0 or len(positive_negative) == 1:
		return 0

	sum = np.sum(sort_data[:, 1])
	positive_count = (np.sum(positive_negative[1][:, 1]) )/ sum
	negative_count = (np.sum(positive_negative[0][:, 1]) )/ sum

	return (1 - negative_count ** 2 - positive_count ** 2)


def calculate_gain_value(label, data, root_gain):

	value = np.empty([len(data), 3])
	value[:, 0] = label[:, 0]
	value[:, 1] = label[:, 1]
	value[:, 2] = data

	gain = 0
	sum = np.sum(value[:, 1])

	val_t = value[value[:, 2] == 1]
	val_f = value[value[:, 2] == 0]

	feature_true = val_t
	feature_false = val_f

	u_left = calculate_gini_index(feature_true[:, 0:2])
	u_right = calculate_gini_index(feature_false[:, 0:2])

	p_left = (np.sum(feature_true[:, 1]))/ sum
	p_right = (np.sum(feature_false[:, 1]))/ sum

	gain = root_gain - p_left * u_left - p_right * u_right

	return gain

def create_adaboost(data, depth, current_depth, tree):
	if current_depth == depth:
		label = predict_label_value(data[:, 0:2])
		tree.insert(None, True, label)
		return

	root_gain = calculate_gini_index(data[:, 0:2])

	gain = 0
	best_feature = 0

	# Calculate the gain for every feature and find the best gain value
	for featureIndex in range(2, data.shape[1]):

		feature_gain = calculate_gain_value(data[:, 0:2], data[:, featureIndex], root_gain)

		# Update the gain
		if feature_gain > gain:
			gain = feature_gain
			best_feature = featureIndex

	# If the gain is zero == leaf node
	if gain == 0:
		label = predict_label_value(data[:, 0:2])
		tree.insert(None, True, label)
		return

	# Data instances where this feature is true
	feature_true = data[data[:,best_feature] == 1]

	# Data instances where this feature is false
	feature_false = data[data[:,best_feature] == 0]

	label = predict_label_value(data[:, 0:2])
	tree.insert(best_feature, False, label)

	tree.left = DecisionTree()
	tree.right = DecisionTree()

	current_depth = current_depth + 1

	# Check if the number of data nodes is zero
	if len(feature_true) > 0:
		create_adaboost(feature_true, depth, current_depth, tree.left)
	if len(feature_false) > 0:
		create_adaboost(feature_false, depth, current_depth, tree.right)


def predict_label_value(data):
	size = len(data)
	# Sort the data based on the final class label
	sort_data = data[np.argsort(data[:, 0])]

	# Split the positive and negative Samples and their corresponding weights into two parts
	positive_negative = np.split(sort_data, np.where(sort_data[:, 0] > 0)[0][:1])

	# Take the sum of the weights for the denominator
	sum = np.sum(sort_data[:, 1])

	positve_total_weight = 0
	negative_total_weight = 0

	if len(positive_negative[0]) >= 1 and positive_negative[0][0, 0] == -1:

		negative_total_weight = (np.sum(positive_negative[0][:, 1])) / sum
		if len(positive_negative) > 1:
			positve_total_weight = (np.sum(positive_negative[1][:, 1])) / sum
	else:
		positve_total_weight = (np.sum(positive_negative[1][:, 1])) / sum

	if positve_total_weight > negative_total_weight:
		prediction = 1
	else:
		prediction = -1

	return prediction


def calculate_error(tree, data, depth):
	error = 0
	sign_values = []
	for row in data:
		prediction = predict_result(tree, row, 0, depth)

		if row[0] != prediction:
			error = error + row[1]
			y_value = 1
		else:
			y_value = -1
		sign_values.append(y_value)
	error = error/ np.sum(data[:, 1])
	return error, sign_values

# Calculate the accuracy of the adaboost algorithm
def accuracy_adaboost(tree_list, alpha_list, data, maxDepth):
	size = len(data)
	error = 0

	# Creating the weight values -- Initial weights are uniformly distributed
	weight_vector = np.empty(size)
	data = np.insert(data, 1, weight_vector, axis = 1)

	for row in data:
		sum = 0
		index = 0
		for tree in tree_list:
			alpha = alpha_list[index]
			prediction = predict_result(tree, row, 0, maxDepth)
			sum = sum + prediction * alpha
			index = index + 1
		if np.sign(sum) != row[0]:
			error = error + 1

	accuracy = ((size - error) / size) * 100
	return accuracy
