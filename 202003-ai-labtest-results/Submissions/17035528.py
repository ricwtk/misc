import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as pt
import numpy as np


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

dt = pd.DataFrame(data, columns=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"])
y = pd.DataFrame(data, columns=["Glass type"])

x_train, x_test, y_train, y_test = train_test_split(dt, y, test_size = 0.3, random_state = 1)

accuracyGraph = []
K = []

for x in range(100):
    
    if ((x != 0) and (x % 2 != 0)):
        
        knc = KNeighborsClassifier(x)
        K.append(x)
        knc = KNeighborsClassifier(x)
        knc.fit(x_train, np.ravel(y_train))
        accuracyGraph.append(knc.score(x_test, y_test))
        print('k =', K, ', accuracy:', accuracyGraph[-1].round(4))

pt.figure()
pt.plot(K, accuracyGraph)
pt.xlabel('K value')
pt.ylabel('Accuracy')
pt.title('Comparison of KNN acccuracy with K')