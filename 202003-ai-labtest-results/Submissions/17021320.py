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
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

x = data.drop(columns=["Glass type"])

y = data["Glass type"].values

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

accuracy_list = []
k_list = []

for i in range(3,100):
    if i % 2 != 0:
        k_list.append(i)

for i in k_list:
    
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    accuracy = knn.score(X_test,y_test)
    accuracy_list.append(round(accuracy,4))


for i,j in zip(k_list,accuracy_list):
    print(i, j)

plt.title("Accuracy against K value")
plt.xlabel("Value of K")
plt.ylabel("Accuracy score")
plt.plot(k_list,accuracy_list)
plt.show()







