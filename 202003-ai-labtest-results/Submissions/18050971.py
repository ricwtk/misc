import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier #import KNN classifier class

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
        'attributes': data.drop('Glass type', axis=1),
        'target': pd.DataFrame(data, columns = ['Glass type']),
        'targetNames': list(data.columns)
        }

#split dataset into 70-30 for train-test proportion
from sklearn.model_selection import train_test_split
for dt in [glass]:
    x_train, x_test, y_train, y_test = train_test_split(dt['attributes'], dt['target'], test_size = 0.2, random_state = 1)
    dt['train'] = {
        'attributes': x_train,
        'target': y_train}
    dt['test'] = {
        'attributes': x_train,
        'target': y_train}
   

#instantiate object with k = 3 for KNC class
knc = KNeighborsClassifier(3)

#train classifier with training data first
#exclude 'ID', 'RI' and 'Glass'
input_columns = glass['attributes'].columns[2:9].tolist()
x_train = glass['train']['attributes'][input_columns]
y_train = glass['train']['target']
knc.fit(x_train, y_train)

#use .predict to predict result of testing data
x_test = glass['test']['attributes'][input_columns]
y_test = glass['test']['target']
y_predict = knc.predict(x_test)

#compare prediction and value of test data
print(pd.DataFrame(list(zip(y_test, y_predict)), columns = ['target', 'predicted']))

#calculate accuracy of prediction
print(f' Accuracy: {knc.score(x_test, y_test):.4f}')

#Visualization using scatterplot
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

k_list = []
accuracy = []
for i in range(3, 100):
    if (i % 2 != 0):
        k_list.append(i)
        knc = KNeighborsClassifier(i)
        knc.fit(x_train, y_train)
        accuracy.append(knc.score(x_test, y_test))

#prepare colormaps
colormap = cm.get_cmap('tab20')
cm_dark = ListedColormap(colormap.colors[::2])
cm_light = ListedColormap(colormap.colors[1::2])



#plot decision boundary
plt.figure()

#plot test and training data
plt.scatter(k_list, accuracy)
plt.xlabel('kvalue')
plt.ylabel('Accuracy')
plt.title('Accuracy of K')
plt.show()

