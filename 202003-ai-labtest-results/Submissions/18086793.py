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

data_input = data.iloc[:, 0:9]
data_target = data.iloc [:, -1]
x_train, x_test, y_train, y_test = train_test_split(data_input, data_target, train_size = 0.7, test_size = 0.3, random_state = 1)

kList = []
accList = []
for i in range(3, 100, 2):
    knc = KNeighborsClassifier(i)
    knc.fit(x_train, y_train)
    accuracy = round(knc.score(x_test, y_test), 4)
    kList.append(i)
    accList.append(accuracy)
    print("\n-------------------------")
    print("k value: " + str(i))
    print(f'Accuracy: {knc.score(x_test,y_test):.4f}')

plt.plot(kList, accList)
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.title('Graph of accuracy against k')









