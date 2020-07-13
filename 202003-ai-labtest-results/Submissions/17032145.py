import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

# Instructions
# 1. split the data into 70% training and 30% testing data
#     - use Na, Mg, Al, Si, K, Ca, Ba, and Fe (i.e. all columns except Glass type) as the input features.
#     - use Glass type as the target attribute.

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)

# print(element)
data = {
  'attributes': pd.DataFrame(data, columns=[ "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]),
  'target': pd.DataFrame(data, columns=['Glass type']),
  'train': None,
  'test': None
}

print(data['attributes'])
for dt in data:
  x_train, x_test, y_train, y_test = train_test_split(data['attributes'], data['target'], test_size=0.3, random_state=1)
  print(y_train)
  data['train'] = {
        'attributes': x_train,
        'target': y_train
        }
  data['test'] = {
        'attributes': x_test,
        'target': y_test
        }
# 2. plot the accuracy of knn classifiers for all odd value of k between 3 to 100, i.e. k = 3, 5, 7, ..., 100. This is achieved by fulfilling the following tasks:
#     i. create a loop to
#       A. fit the training data into knn classifiers with respective k.
#       B. calculate the accuracy of applying the knn classifier on the testing data.
#       C. print out the accuracy for each k.
#
#     ii. plot a line graph with the y-axis being the accuracy for the respective k and x-axis being the value of k. You DO NOT need to save the graph.


# start your code after this line