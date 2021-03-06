import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)

''' Instructions
1. split the data into 70% training and 30% testing data
	- use Na, Mg, Al, Si, K, Ca, Ba, and Fe (i.e. all columns except Glass type) as the input features.
	- use Glass type as the target attribute.

2. plot the accuracy of knn classifiers for all odd value of k between 3 to 100, i.e. k = 3, 5, 7, ..., 100. This is achieved by fulfilling the following tasks:
	i. create a loop to 
	  A. fit the training data into knn classifiers with respective k.
	  B. calculate the accuracy of applying the knn classifier on the testing data.
	  C. print out the accuracy for each k.

	ii. plot a line graph with the y-axis being the accuracy for the respective k and x-axis being the value of k. You DO NOT need to save the graph.
'''

# start your code after this line
data = {
	'attributes': data[["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]],
	'target': pd.DataFrame(data['Glass type'])
}

x_train, x_test, y_train, y_test = train_test_split(data['attributes'], data['target'], test_size=0.3, random_state=1)
data['train'] = {
	'attributes': x_train,
	'target': y_train
}
data['test'] = {
	'attributes': x_test,
	'target': y_test
}

k_values = []
accuracies = []

for i in range(3, 100, 2):
	knc = KNeighborsClassifier(i)

	input_columns = data['attributes'].columns.tolist()
	x_train = data['train']['attributes'][input_columns]
	y_train = data['train']['target']['Glass type']
	knc.fit(x_train, y_train)

	x_test = data['test']['attributes'][input_columns]
	y_test = data['test']['target']['Glass type']
	y_predict = knc.predict(x_test)

	accuracy = knc.score(x_test, y_test)

	k_values.append(i)
	accuracies.append(accuracy)
	print(f'Accuracies for k with value of {i} is {accuracy:.4f}')

plt.figure(figsize=[12,8])
plt.plot(k_values, accuracies)
plt.xlabel('k values')
plt.ylabel('Accuracy')
# plt.show()