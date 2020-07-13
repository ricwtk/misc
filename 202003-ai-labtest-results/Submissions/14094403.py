import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

# 1
# Split data set into input features (X) and target attribute (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, 9].values

# Split data into testing and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 2i
k_start = 3
k_stop = 100
k_step = 2
accuracy_result = []
for k in range(k_start, k_stop, k_step):
    # Train
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Make predictions using test data
    y_predict = knn.predict(X_test)

    # Append mean of error for all the predicted values
    accuracy = accuracy_score(y_test, y_predict)
    print('The accuracy for k = ' + str(k) + ' is: ' + str(accuracy))
    accuracy_result.append(accuracy)
    # error.append(np.mean(predictions != y_test))

# 2ii
plt.figure(figsize=(12, 6))
plt.plot(range(k_start, k_stop,k_step), accuracy_result, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy vs k-Value')
plt.xlabel('k-Value')
plt.ylabel('Accuracy')
plt.show()

