import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import sys
import matlpotlib.pyplot as plt
from sklearn import linear_model
from math import fabs, inf
from sklearn.neighbors import KNeighborsClassifier

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
for k in range(3, len(x_train)-99):
glass={
  "attributes": pd.DataFrame(data,columns=["Na","Mg","Al","Si","K","Ca","Ba","Fe"]),
  'target':pd.DataFrame(data,columns=["Glass type"]),
  }
X_train, X_test, y_train, y_test = train_test_split(glass["attributes"], glass["target"], test_size=0.3)
glass['train'] = {
    'attributes': x_train,
    'target': y_train
  }
  glass['test'] ={
  'attributes': x_test,
  'target': y_test
  }
  input_columns = data['attributes'].columns[:8].tolist()
x_train = data['train']['attributes'][input_columns]
y_train = data['train']['target']
knc.fit(x_train, y_train)

print(pd.DataFrame(list(zip(y_test,y_predict)), columns=['target', 'predicted']))


regrmodel = linear_model.LinearRegression()
regrmodel.fit(X_train[['attributes']], y_train['target'])
y_train_pred = regrmodel.predict(X_train[['attributes']])
y_test_pred = regrmodel.predict(X_test[['attributes']])

y_train_pred = pd.Series(y_train_pred)
y_train_pred.index = y_train.index

y_test_pred = pd.Series(y_test_pred)
y_test_pred.index = y_test.index



traincost = cost(y_train['target'], y_train_pred)
testcost = cost(y_test['target'], y_test_pred)


plt.figure()
plt.scatter(X_train['attributes'], y_train['target'], color='red')
plt.plot(X_train['k'], y_train_pred, '-', color='green')
plt.title('Training data (sklearn)')
plt.xlabel('k')

plt.figure()
plt.scatter(X_test['attributes'], y_test['target'], color='red')
plt.plot(X_test['K'], y_test_pred, '-', color='green')
plt.title('data (sklearn)')
plt.xlabel('k')


