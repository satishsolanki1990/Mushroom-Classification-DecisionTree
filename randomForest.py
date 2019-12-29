import numpy as np
from decision_tree import DecisionTree
from dt_implementation import decision_tree, predict_result

# Create a random forest
def create_random_forest(m, n, d, data):
	length = len(data)
	tree_list = []
	
	for i in range(n):
		tree_list.append(DecisionTree())

		# Sample with replacement training data
		sample = np.random.choice(length, length, replace = True)
		sample = data[sample]
		decision_tree(sample, d, 0, tree_list[i], m)

	return tree_list

# Calculate the accuracy for the random forest
def accuracy_random_forest(list_of_trees, data, depth):
	sample_size = len(data)
	error = 0

	for row in range(0, sample_size):

		count_positive = 0
		count_negative = 0

		# Calculate the prediction value for all the trees in the forest and find the count of
		# predictions for all the trees

		for tree in list_of_trees:
			prediction = predict_result(tree, data[row, :], 0, depth)
			if prediction == 1:
				count_positive = count_positive + 1
			else:
				count_negative = count_negative + 1

		# Calculate the total count for the final prediction
		if count_positive > count_negative:
			final_prediction = 1
		else:
			final_prediction = -1
		if final_prediction != data[row, 0]:
			error += 1
		accuracy = ((sample_size - error) * 1.0/ sample_size) * 100

	return accuracy