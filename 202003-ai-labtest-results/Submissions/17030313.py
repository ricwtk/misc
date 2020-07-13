import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"],
                   index_col=0)

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
glass = {
    'attributes': data.drop('Glass type', axis=1),
    'target': pd.DataFrame(data, columns=['Glass type']),
    'targetNames': list(data.columns)
}

x_train, x_test, y_train, y_test = train_test_split(glass['attributes'], glass['target'], test_size=0.3, random_state=1)

glass['train'] = {
    'attributes': x_train,
    'target': y_train
}
glass['test'] = {
    'attributes': x_test,
    'target': y_test
}

ks = []
accuracies = []

for k in range(3, 101):
    if k % 2 == 1:
        knc = KNeighborsClassifier(k)

        cols = glass['attributes'].columns.tolist()

        x_train = glass['train']['attributes'][cols]
        y_train = glass['train']['target'].values.ravel()
        knc.fit(x_train, y_train)

        x_test = glass['test']['attributes'][cols]
        y_test = glass['test']['target'].values.ravel()
        y_predict = knc.predict(x_test)

        # Output
        print(f'k = {k}')
        print(pd.DataFrame(list(zip(y_test, y_predict)), columns=['target', 'predicted']))
        print(f'Accuracy: {knc.score(x_test, y_test):.4f}')

        ks.append(k)
        accuracies.append(knc.score(x_test, y_test))

plt.figure(figsize=[12, 8])
plt.title('Classification: k vs Accuracy')

plt.ylim(ymin=0)

plt.scatter(ks, accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')

# plt.waitforbuttonpress()
