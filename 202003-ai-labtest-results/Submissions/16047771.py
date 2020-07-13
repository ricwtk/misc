import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import glass.csv as DataFrame

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

glass = datasets.load_glass.csv()
dt = pd.DataFrame(glass.data, columns=data.names),
y = pd.DataFrame(glass.target, colummns=['target'])

X_train, X_test, y_train, y_test = train_test_split(dt, y, test_size=0.3)

def model(x, m, c):
  return m*x + c

def cost(y, yh):
  return ((y-yh)**2).mean()

def derive(x, y, yh):
  return {
    'm': ((y-yh)*x).mean()*-2,
    'c': (y-yh).mean()*-2
  }

#sir its too hard :(
  



    