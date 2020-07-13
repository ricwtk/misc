import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
glass = {
    'attributes': data.iloc[:,0:9],
    'target' : data.iloc[:,-1],
    'targetNames': "Glass Type"
}

x_train, x_test, y_train, y_test = train_test_split(glass['attributes'], glass['target'], test_size=0.3, random_state=1)
glass['train'] = {
    'attributes': x_train,
    'target': y_train
}
glass['test'] = {
    'attributes': x_test,
    'target': y_test
}

input_columns = glass['attributes'].columns[:9].tolist()
x_train = glass['train']['attributes'][input_columns]
y_train = glass['train']['target']
x_test = glass['test']['attributes'][input_columns]
y_test = glass['test']['target']


i = 3
accuracy_list = []
k_number_list = []

while i <= 100:
    knc = KNeighborsClassifier(i)
    knc.fit(x_train, y_train)
    y_predict = knc.predict(x_test)


    k_number_list.append(i)
    accuracy = knc.score(x_test,y_test)
    accuracy_list.append(accuracy)
    print("Number of k:",i,"Accuracy: ",accuracy)
    i += 2

plt.plot(k_number_list,accuracy_list)
plt.xlabel("Number of K")
plt.ylabel("Accuracy")
plt.show()
