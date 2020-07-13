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

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap


x = pd.DataFrame(data, columns=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"])
y = pd.DataFrame(data, columns=["Glass type"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

colormap = cm.get_cmap('tab20')
cm_dark = ListedColormap(colormap.colors[::2])
cm_light = ListedColormap(colormap.colors[1::2])

k = 3
k_list = []
accuracy_list = []

while k <= 100:
  k_list.append(k)
  knc = KNeighborsClassifier(k)
  knc.fit(x_train, y_train)
  print("Accuracy for k = {} is {}".format(k, knc.score(x_test, y_test)))
  accuracy_list.append(knc.score(x_test, y_test))
  k += 2

plt.figure()
plt.scatter(k_list, accuracy_list)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy for each k')
plt.show()