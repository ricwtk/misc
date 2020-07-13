from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from math import fabs, inf
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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

dt = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['Glass type'])

X_train, X_test, y_train, y_test = train_test_split(dt, y, test_size=0.2)

def model(x, m, c):
  return m*x + c

def cost(y, yh):
  return ((y-yh)**2).mean()

def derivatives(x, y, yh):
  return {
    'm': ((y-yh)*x).mean()*-2,
    'c': (y-yh).mean()*-2
  }

# initial values
learningrate = 0.1
m = []
c = []
J = []
m.append(0)
c.append(0)
J.append(cost(y_train['Glass type'], X_train['data'].apply(lambda x: model(x, m[-1], c[-1]))))

# termination conditions
J_min = 0.01
del_J_min = 0.0001
max_iter = 10000

def getdelJ():
  if len(J) > 1:
    return fabs(J[-1] - J[-2])/J[-1]
  else:
    return inf

plt.ion()

plt.scatter(X_train['data'], y_train['Glass type'], color='red')
plt.title('Training data')
plt.xlabel('Data Glass')
line = None
while J[-1] > J_min and getdelJ() > del_J_min and len(J) < max_iter:
  der = derivatives(X_train['data'], y_train['Glass type'], X_train['data'].apply(lambda x: model(x, m[-1], c[-1])))
  m.append(m[-1] - learningrate * der['m'])
  c.append(c[-1] - learningrate * der['c'])
  J.append(cost(y_train['Glass type'], X_train['data'].apply(lambda x: model(x, m[-1], c[-1]))))

  print('.', end='')
  sys.stdout.flush()

  if line:
    line[0].remove()
  line = plt.plot(X_train['data'], X_train['data'].apply(lambda x: model(x, m[-1], c[-1])), '-', color='green')
  plt.pause(0.001)

y_train_pred = X_train['data'].apply(lambda x: model(x, m[-1], c[-1]))
y_test_pred = X_test['data'].apply(lambda x: model(x, m[-1], c[-1]))
print('\nAlgorithm terminated with')
print(f'  {len(J)} iterations')
print(f'  m {m[-1]}')
print(f'  c {c[-1]}')
print(f'  training cost {J[-1]}')
testcost = cost(y_test['target'], y_test_pred)
print(f'  testing cost {testcost}')

plt.figure()
plt.scatter(X_test['data'], y_test['Glass type'], color='red')
plt.plot(X_test['data'], y_test_pred, '-', color='green')
plt.title('Testing data')

##### sklearn
regrmodel = linear_model.LinearRegression()
regrmodel.fit(X_train[['data']], y_train['Glass type'])
y_train_pred = regrmodel.predict(X_train[['data']])
y_test_pred = regrmodel.predict(X_test[['data']])

y_train_pred = pd.Series(y_train_pred)
y_train_pred.index = y_train.index

y_test_pred = pd.Series(y_test_pred)
y_test_pred.index = y_test.index

print('sklearn')
print(f'  m {regrmodel.coef_}')
print(f'  c {regrmodel.intercept_}')
traincost = cost(y_train['target'], y_train_pred)
testcost = cost(y_test['target'], y_test_pred)
print(f'  training cost: {traincost}')
print(f'  testing cost: {testcost}')

plt.figure()
plt.scatter(X_train['data'], y_train['Glass type'], color='red')
plt.plot(X_train['data'], y_train_pred, '-', color='green')
plt.title('Training data (sklearn)')
plt.xlabel('Data glass')

plt.figure()
plt.scatter(X_test['data'], y_test['Glass type'], color='red')
plt.plot(X_test['data'], y_test_pred, '-', color='green')
plt.title('Testing data (sklearn)')
plt.xlabel('Data glass')


##### multivariate linear regression
regrmodel = linear_model.LinearRegression()
regrmodel.fit(X_train[['data','bp']], y_train['Glass type'])
y_train_pred = regrmodel.predict(X_train[['data','bp']])
y_test_pred = regrmodel.predict(X_test[['data','bp']])

y_train_pred = pd.Series(y_train_pred)
y_train_pred.index = y_train.index

y_test_pred = pd.Series(y_test_pred)
y_test_pred.index = y_test.index

print('sklearn')
print(f'  m {regrmodel.coef_}')
print(f'  c {regrmodel.intercept_}')
traincost = cost(y_train['target'], y_train_pred)
testcost = cost(y_test['target'], y_test_pred)
print(f'  training cost: {traincost}')
print(f'  testing cost: {testcost}')

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['data'], X_train['bp'], y_train['target'], color='red')
ax.scatter(X_train['data'], X_train['bp'], y_train_pred, color='green')
ax.set_xlabel('Data glass')
ax.set_ylabel('Type of glass output')
ax.set_title('Training data (sklearn multivariate)')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['data'], X_test['k'], y_test['target'], color='red')
ax.scatter(X_test['data'], X_test['k'], y_test_pred, color='green')
ax.set_xlabel('Data glass')
ax.set_ylabel('Type of glass output')
ax.set_title('Testing data (sklearn multivariate)')


plt.ioff()
plt.show()
  