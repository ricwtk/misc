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

import numpy as np
import matplotlib.pyplot as plt
import vis
from skfuzzy import control as ctrl
from skfuzzy import membership as mf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data['Glass type'] = data.target

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.7)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

vis.vis2d(ax, mlp, X_train, y_train, X_test, y_test)

fig = plt.figure()
axes = vis.vis3d(fig, mlp, X_train, y_train, X_test, y_test)
for i,a in enumerate(axes):
  a.set_title(iris.target_names[i])
  a.set_xticklabels([])
  a.get_yaxis().set_visible(False)
axes[-1].set_xticklabels(iris.feature_names)