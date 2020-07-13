import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

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

dt = data[['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y = data['Glass type']
x_train,x_test,y_train,y_test = train_test_split(dt,y,test_size=0.3)

## function to take input of data and number of clusters, return centroids and other data
def get_random_centroids(data_points, n_centroids=2):
  pass

## function to group data according to centroids
def group_to_centroids(data_points, centroids):
  pass

## function to calculate centroids from grouped data
def find_centroids(data_points, groups):
  pass

from sklearn.datasets import make_blobs
data = make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=1.6, random_state=50)
points = data[0]
import matplotlib.pyplot as plt
plt.scatter(data[0][:,0], data[0][:,1], c=data[1])

# identify initial centroids
centroids, others = get_random_centroids(points, 4)

while points < 100:
    groups = group_to_centroids(others, centroids)
    centroids = find_centroids(others, groups)
    
print('terminated')

