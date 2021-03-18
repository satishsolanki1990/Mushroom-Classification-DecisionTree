import numpy as np
from random import sample, randint
import data_process as prep
from dt_implementation import decision_tree, accuracy_decision_tree, test_decision_tree
from decision_tree import DecisionTree
from randomForest import create_random_forest, accuracy_random_forest
from adaboost import adaboost, accuracy_adaboost
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

red_patch = mpatches.Patch(color='red', label='Training')
blue_patch = mpatches.Patch(color='blue', label='Validation')
plt.legend(handles=[red_patch, blue_patch])

if __name__ == '__main__':

	print("Options:")
	print("1. Decision Tree")
	print("2. Random Forest")
	print("3. AdaBoost")
	print("4. Exit")

	while(1):

		trainData = prep.fileRead('train.csv')	# Read Training Examples
		trainData = prep.changeData(trainData)
		validData = prep.fileRead('val.csv')	# Read Validation Data
		validData = prep.changeData(validData)
		testData = prep.fileRead('test.csv')
		testData = prep.changeData(testData)

		TrainingIterations = np.array([])
		ValidationIterations = np.array([])

		choice = int(input("Please enter your choice: "))

		# Decision Tree Classifier
		if choice == 1:

			maximum_Depth = 8
			tree = DecisionTree()

			decision_tree(trainData, maximum_Depth, 0, tree, 100)

			train_acc_list = []
			valid_acc_list = []
			for i in range (1, maximum_Depth):

				train_acc_list.append(accuracy_decision_tree(tree, trainData, i))
				valid_acc_list.append(accuracy_decision_tree(tree, validData, i))

			print("Printing Training accuracy for Decision Tree")
			print(train_acc_list)
			# TrainingIterations = np.append(maximum_Depth, train_acc_list)
			print("Printing Validation accuracy for Decision Tree ")
			print(valid_acc_list)
			# ValidationIterations = np.append(maximum_Depth, valid_acc_list)

			predicted_values = test_decision_tree(tree, testData, maximum_Depth)
			print ("The sum is ",  np.sum (predicted_values ))
			print("Decision tree predicted values: ", predicted_values)
			np.savetxt("predicted_values.txt", predicted_values)

			# plt.title("Accuracy versus Depth", loc='center')
			# plt.plot(TrainingIterations, color='red')
			# plt.plot(ValidationIterations, color="blue")
			# plt.ylabel('Accuracy')
			# plt.xlabel('Depth')
			# plt.savefig('test.png')

		# Random Forest Classifier
		elif choice == 2:

			d = 2
			n = 20
			m = 35

			train_acc_list = []
			valid_acc_list = []
			itr_list = []

			for i in range(0, 20):

				itr_list.append(n)
				list_of_trees = create_random_forest(m, n, d, trainData)

				train_acc_list.append(accuracy_random_forest(list_of_trees, trainData, d))
				valid_acc_list.append(accuracy_random_forest(list_of_trees, validData, d))

				# print("Training Accuracy for Random Forest ")
				TrainingIterations = np.append(i, train_acc_list)
				print(train_acc_list)
				# print("Validation Accuracy for Random Forest ")
				ValidationIterations = np.append(i, valid_acc_list)
				print(valid_acc_list)

			# plt.title("Accuracy versus Iterations", loc='center')
			# plt.plot(TrainingIterations, color='red')
			# plt.plot(ValidationIterations, color="blue")
			# plt.ylabel('Accuracy')
			# plt.xlabel('Iterations')
			# plt.savefig('test.png')

		# AdaBoost Algorithm
		elif choice == 3:

			train_acc_list = []
			valid_acc_list = []
			itr_list = []
			depth = 2

			for l in [6]:		# [1,2,5,10,15,25]
				itr_list.append(l)
				tree_list, alpha_list = adaboost(trainData, l, depth)

				train_acc_list.append(accuracy_adaboost(tree_list, alpha_list, trainData, depth))
				valid_acc_list.append(accuracy_adaboost(tree_list, alpha_list, validData, depth))

			# print ("Training Accuracy for AdaBoost")
			print(train_acc_list)
			# print ("Validation Accuracy for AdaBoost")
			print(valid_acc_list)

			# plt.title("Accuracy versus m", loc='center')
			# plt.plot(TrainingIterations, color='red')
			# plt.plot(ValidationIterations, color="blue")
			# plt.ylabel('Accuracy')
			# plt.xlabel('m')
			# plt.savefig('test.png')

		# Exit option
		else:
			print("Exiting.")
			exit(0)
