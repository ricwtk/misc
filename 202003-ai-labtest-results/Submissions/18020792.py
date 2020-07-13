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


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

input_columns = data.columns[:-1].tolist()
dt = data[input_columns]
y = data['Glass type']
X_train, X_test, y_train, y_test = train_test_split(dt, y, test_size=0.3)


accuracy = []
k1 = list(range(3,100,2))
for elem in k1:
    knc = KNeighborsClassifier(elem) 

    knc.fit(X_train, y_train)   

    y_predict = knc.predict(X_test)
    
    acc = knc.score(X_test,y_test)
    print('k=',elem, f'Accuracy: {acc:.4f}')
    accuracy.append(acc)
    
#classification
plt.figure()
plt.title("kNN Classification")
plt.plot(k1, accuracy)
plt.xlabel("k")
plt.ylabel("Accuracy")

