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

# splitting the data into training and testing
glass = {
  'attributes' : pd.DataFrame(data, columns=["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]),
  'target' : pd.DataFrame(data, columns=["Glass type"])
}

x_train, x_test, y_train, y_test = train_test_split(glass['attributes'], glass['target'], test_size = 0.3 ,random_state = 1)
glass['train'] = {
  'attributes': x_train,
  'target' : y_train
}

glass['test'] = {
  'attributes' : x_test,
  'target' : y_test
}

# classification 
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(3)

input_columns = glass['attributes'].columns.tolist()
x_train = glass['train']['attributes'][input_columns]
y_train = glass['train']['target']
knc.fit(x_train, y_train)

x_test = glass['test']['attributes']
y_test = glass['test']['target']
y_predict = knc.predict(x_test)

print(pd.DataFrame(list(zip(y_test, y_predict)), columns=['target', 'SSpredicted']))
print(f'Accuracy: {knc.score(x_test,y_test):.4f}')

# visualisation
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

## colormaps
colormap = cm.get_cmap('tab20')
cm_dark = ListedColormap(colormap.colors[::2])
cm_light = ListedColormap(colormap.colors[1::2])

## calculate decision boundaries
import numpy as np
x_min = glass['attributes'][input_columns[0]].min()
x_max = glass['attributes'][input_columns[0]].max()
x_range = x_max - x_min

x_min = x_min - 0.1 * x_range
x_max = x_max + 0.1 * x_range

y_min = glass['attributes'][input_columns[1]].min()
y_max = glass['attributes'][input_columns[1]].max()
y_range = y_max - y_min

y_min = y_min - 0.1 * y_range
y_max = y_max + 0.1 * y_range
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01*x_range), np.arange(y_min, y_max, .01*y_range))
z = knc.predict(list(zip(xx.ravel(), yy.ravel())))
z = z.reshape(xx.shape)

## display decision boundary
plt.figure(figsize=[12,8])
plt.pcolormesh(xx, yy, z, cmap=cm_light)

## plot training and testing data
plt.scatter(x_train[input_columns[0]], x_train[input_columns[1]], c=y_train, label='Training data', cmap=cm_dark, edgecolor='black', linewidth=1, s=150)
plt.scatter(x_test[input_columns[0]], x_test[input_columns[1]], c=y_test, marker='*', label='Testing data', cmap=cm_dark, edgecolor='black', linewidth=1, s=150)

## label the graph
plt.xlabel(input_columns[0])
plt.ylabel(input_columns[1])
plt.legend()

k_list = []
accuracy_list = []
i = 3
for i in range(100):
  k_list.append(i)
  i += 2

for k in k_list:
  knc = KNeighborsClassifier(k)
  knc.fit(x_train, y_train)
  accuracy_list.append(knc.score(x_test, y_test))
  print(pd.DataFrame(list(zip(y_test, y_predict)), columns=['target', 'predicted']))
  print(f'Accuracy: {knc.score(x_test,y_test):.4f}')

plt.figure()
plt.scatter(k_list, accuracy_list)
plt.xlabel('$k$')
plt.ylabel('Accuracy')
plt.title('Comparison of accuracy for different k')
plt.show()

