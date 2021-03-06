from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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


y = pd.DataFrame(data, columns=['Glass type'])
x_train, x_test, y_train, y_test = train_test_split(
    data.iloc[:, 1:9], y, test_size=0.3)



k_values = np.arange(3, 101, 2)
k_values = pd.Series(k_values)
accuracies = np.arange(0, 49, 1, dtype=float)

for index, k in enumerate(k_values):

    knc = KNeighborsClassifier(k)
    
    # test = y_train['Glass type']
    knc.fit(x_train, y_train.values.ravel())
        
    accuracies[index] = knc.score(x_test, y_test)
    
    print(f'KNN: {k}, Accuracy: {knc.score(x_test,y_test):.4f}')


pt.figure()
pt.plot(k_values, accuracies)
pt.title('Prediction Accuracy')
pt.xlabel('K')
pt.ylabel('accuracy')