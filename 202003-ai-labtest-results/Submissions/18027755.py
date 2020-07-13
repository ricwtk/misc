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

data = {
        'attributes': pd.DataFrame(data, columns=["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]),
        'target': pd.DataFrame(data, columns=["Glass type"])
        }

x_train, x_test, y_train, y_test = train_test_split(data['attributes'], data['target'], test_size=0.3)
knc_accuracy = []
k_iterations = []
for i in range(3,100,2):
    knc = KNeighborsClassifier(i)
    knc.fit(x_train, y_train)
    y_predict = knc.predict(x_test)
    print(pd.DataFrame(list(zip(y_test,y_predict)), columns=['target', 'predicted']))
    print('\nCurrent k-value:', i)
    print(f'Accuracy: {knc.score(x_test,y_test):.4f}')
    print('')
    knc_accuracy.append(knc.score(x_test,y_test))
    k_iterations.append(i)
    
plt.figure()
plt.plot(k_iterations, knc_accuracy)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('KNN Classification')