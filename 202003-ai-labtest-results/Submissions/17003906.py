"""
Lab Test
(AI CSC3206 Semester March 2020)

Name: Leong Wen Hao 
Student ID: 17003906
"""

import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as pt

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

attribute_columns = ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
target_columns = ["Glass type"]
data  = {
  'attributes': pd.DataFrame(data, columns = attribute_columns),
  'target': pd.DataFrame(data, columns = target_columns)
}

#split the data for test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data['attributes'], data['target'], test_size=0.3)

data['train'] = {
                'attributes': x_train,
                'target': y_train
            }

data['test'] = {
                'attributes': x_test,
                'target': y_test
            }

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(5)

input_columns = data['attributes'].columns[:2].tolist()
x_train = data['train']['attributes'][input_columns]
y_train = data['train']['target']
knc.fit(x_train, y_train)

x_test = data['test']['attributes'][input_columns]
y_test = data['test']['target']
y_predict = knc.predict(x_test)

k_values = []
accuracies = []

start = 3
stop = 100
step = 2
for k in range(start, stop, step):

    k_values.append(k)
    knc = KNeighborsClassifier(k)
    knc.fit(x_train, y_train)
    accuracy = knc.score(x_test, y_test)
    accuracies.append(accuracy)

    print("\nK: {} \nAccuracy: {}".format(k, accuracy))

pt.figure()
pt.plot(k_values, accuracies)

#labels
pt.xlabel("k")
pt.ylabel("accuracy")
pt.title("accuracy vs k")

pt.show()