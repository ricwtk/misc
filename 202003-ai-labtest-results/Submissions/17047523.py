import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
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

n_samples, n_features = data.shape

X = data.drop(columns=['Glass type']).values ##Drop the "Glass type"
y = data['Glass type'].values ##set as target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

k = 3
while k<=100:
    if k%2 != 0:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train,y_train)
        y_predict = knn.predict(X_test)
        print(pd.DataFrame(list(zip(y_test,y_predict)), columns=['target', 'predicted']))
        print(f'Accuracy: {knn.score(X_test,y_test):.4f}')
    k += 1

##plot    ##Unable to complete
#colormap = cm.get_cmap('tab20')
#cm_dark = ListedColormap(colormap.colors[::2])
#cm_light = ListedColormap(colormap.colors[1::2])    
#
#x_min = X.min()
#x_max = X.max()
#x_range = x_max - x_min
#x_min = x_min - 0.1 * x_range
#x_max = x_max + 0.1 * x_range
#y_min = y.min()
#y_max = y.max()
#y_range = y_max - y_min
#y_min = y_min - 0.1 * y_range
#y_max = y_max + 0.1 * y_range
#
#xx, yy = np.meshgrid(np.arange(x_min, x_max, .01*x_range), 
#                    np.arange(y_min, y_max, .01*y_range))
#z = knn.predict(list(zip(xx.ravel(), yy.ravel())))
#z = z.reshape(xx.shape)
#plt.figure
#plt.pcolormesh(xx, yy, z, cmap=cm_light)
#plt.scatter(X_train[input_columns[0]], X_train[input_columns[1]], 
#            c=y_train, label='Training data', cmap=dia_cm, 
#            edgecolor='black', linewidth=1, s=150)
#plt.scatter(X_test[input_columns[0]], X_test[input_columns[1]], 
#            c=y_test, marker='*', label='Testing data', cmap=dia_cm,
#            edgecolor='black', linewidth=1, s=150)
#plt.xlabel(input_columns[0])
#plt.ylabel(input_columns[1])
#plt.legend()
#plt.colorbar()