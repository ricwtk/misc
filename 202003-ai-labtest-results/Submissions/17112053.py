from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

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

x = pd.DataFrame(data, columns = ['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
y = pd.DataFrame(data, columns = ['Glass type'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

kmean = []
accuracy = []
for n in range (3, 100, 2):
    # Instantiate an object of KNeighborsClassifier class
    knc = KNeighborsClassifier(n)
    
    # Train the classifier with the training data
    knc.fit(x_train, y_train.values.ravel())
    
    # .predict function is used to predict the species of the testing data
    y_predict = knc.predict(x_test)
    
    kmean.append(n)
    accuracy.append(knc.score(x_test, y_test))

print('kmean: ', kmean)
print('accuracy: ', accuracy)

plt.plot(kmean, accuracy, label = "classification")
plt.xlabel('k-mean') 
plt.ylabel('accuracy') 
plt.title('Comparison of k value and accuracy')
plt.legend()
