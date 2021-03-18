import numpy as np

def fileRead(fileName):
	data = np.genfromtxt(fileName, delimiter = ',')

	# Drop the first row of class labels
	data = np.delete(data, 0, axis=0)

	return data


def changeData(data):

	# Insert the class labels as the first column
	data = np.insert(data, 0, data[:, -1], axis=1)

	# Drop the last column of class labels
	data = np.delete(data, -1, axis=1)

	# Change the column labels to -1 and 1
	data[:, 0] = np.where(data[:, 0] == 0, -1, 1)

	return data
