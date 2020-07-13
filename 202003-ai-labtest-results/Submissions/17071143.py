import pandas as pd

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

from sklearn.model_selection import train_test_split
from sklearn import datasets

glass = {
    'attribute': data.drop('Glass type', axis=1),
    'target': pd.DataFrame(data, columns=['Glass type']),
    'targetNames': list(data.columns)
}

x_train, x_test, y_train, y_test = train_test_split(glass['attribute'], glass['target'], test_size=0.3, random_state=1)

glass['train'] = {
    'attribute': x_train,
    'target': y_train
}
glass['test'] = {
    'attribute': x_test,
    'target': y_test
}

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(5)

input_columns = glass['attribute'].columns[:2].tolist()
x_train = glass['train']['attribute'][input_columns]
y_train = glass['train']['target']
knc.fit(x_train, y_train)

x_test = glass['test']['attribute'][input_columns]
y_test = glass['test']['target']
y_predict = knc.predict(x_test)

print(pd.DataFrame(list(zip(y_test, y_predict)), columns=['target', 'predicted']))
print(f'Accuracy: {knc.score(x_test,y_test):.4f}')

# OUTPUT: 
#        target       predicted
# 0  Glass type          6
# Accuracy: 0.4923


#----------------------------------------------------------------------------
import matplotlib.pyplot as plt

k_list = []
accuracy_list = []
for k in range(3, 100):
  k_list.append(k)
  knc = KNeighborsClassifier(k)
  knc.fit(x_train, y_train)
  accuracy_list.append(knc.score(x_test, y_test))


plt.figure()
plt.scatter(k_list, accuracy_list)
plt.xlabel('$k$')
plt.ylabel('Accuracy')
plt.title('Comparison of accuracy for different k')
# plt.show()