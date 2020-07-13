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
x = ['Id', 'RI']
y = [ 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
y = [13.64, 4.49, 1.1,	71.78,	0.06,	8.75,	0,	0,	1]
from sklearn.neighbours import KNeighboursClassifier
knc = KNeighboursClassifier (7)
input_columns = iris['attributes'].columns[:2].tolist()
x_train = iris['train']['attributes'][input_columns]
y_train = iris['train']['target'].species
knc.fit(x_train, y_train)
x_test = iris['test']['attributes'][input_columns]
y_test = iris['test']['target'].species
y_predict = knc.predict(x_test)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)

for i in range(3, 100, 5):
    knc = KNeighborsClassifier(i)
    knc.fit(x_train, y_train)

knc.score(x_test,y_test)
