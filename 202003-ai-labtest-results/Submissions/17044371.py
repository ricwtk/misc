import pandas as pd

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)

data = {
    'attributes': data.iloc[:,0:9],
    'target': data.iloc[:,-1]
    }

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
x_train, x_test, y_train, y_test = train_test_split(data['attributes'], data['target'], test_size=0.3, random_state=1)
data['train'] = {
  'attributes': x_train,
  'target': y_train
}
data['test'] = {
  'attributes': x_test,
  'target': y_test
}


from sklearn.neighbors import KNeighborsClassifier

k=3
input_columns = data['attributes'].columns[:2].tolist()
x_train = data['train']['attributes'][input_columns]
y_train = data['train']['target']


x_test = data['test']['attributes'][input_columns]
y_test = data['test']['target']

arr = [k]

while k <= 100:
    arr.append(k)
    knc = KNeighborsClassifier(k)
    
    knc.fit(x_train, y_train)
    
    y_predict = knc.predict(x_test)
    
    print(f'Accuracy: {knc.score(x_test,y_test):.4f}')
    
    k+=2

import matplotlib.pyplot as plt

import numpy as np

x_min = data['attributes'][input_columns[0]].min()
x_max = data['attributes'][input_columns[0]].max()
x_range = x_max - x_min
x_min = x_min - 0.1 * x_range
x_max = x_max + 0.1 * x_range
y_min = data['attributes'][input_columns[1]].min()
y_max = data['attributes'][input_columns[1]].max()
y_range = y_max - y_min
y_min = y_min - 0.1 * y_range
y_max = y_max + 0.1 * y_range
# xx, yy = np.meshgrid(np.arange(x_min, x_max, .01*x_range), 
#                     np.arange(y_min, y_max, .01*y_range))

# plt.xlabel(input_columns[0])
# plt.ylabel(input_columns[1])
# plt.legend()

accuracy_classification = []

for i in arr:
    knn = KNeighborsClassifier(i)
    knn.fit(x_train, y_train)
    predicted_y = knn.predict(x_test)
    accuracy_classification.append(knn.score(x_test,y_test))
    
    print(f'KNN: {i}, Accuracy: {knn.score(x_test,y_test):.4f}')
    # print(pd.DataFrame(list(zip(y_test,predicted_y)), columns=['target', 'predicted']))
    
plt.figure(figsize=[12,8])

plt.plot(arr, accuracy_classification, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)

plt.title("Accuracy of prediction (Classification)")
plt.xlabel("K")
plt.ylabel("Accuracy")