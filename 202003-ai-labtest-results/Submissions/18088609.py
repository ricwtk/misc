import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np



# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)



x = data.drop(columns=["RI","Glass type"])
y = data["Glass type"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


loop = 3
while loop<100:
  mod = loop % 2
  if mod > 0:
    knn = KNeighborsClassifier(n_neighbors=loop)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print("K",loop,"-",knn.score(x_test, y_test))
  loop = loop + 1


def odd(n):
   return np.arange(1, 2*n, 2)
   
no_neighbors = odd(100)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(no_neighbors,  test_accuracy, label = 'Testing Accuracy')
plt.plot(no_neighbors, test_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()




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