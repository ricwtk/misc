''' Instructions
1. split the data into 70% training and 30% testing data
    - use Na, Mg, Al, Si, K, Ca, Ba, and Fe (i.e. all columcns except Glass type) as the input features.
    - use Glass type as the target attribute.

2. plot the accuracy of knn classifiers for all odd value of k between 3 to 100, i.e. k = 3, 5, 7, ..., 100. This is achieved by fulfilling the following tasks:
    i. create a loop to 
      A. fit the training data into knn classifiers with respective k.
      B. calculate the accuracy of applying the knn classifier on the testing data.
      C. print out the accuracy for each k.

    ii. plot a line graph with the y-axis being the accuracy for the respective k and x-axis being the value of k. You DO NOT need to save the graph.
'''

# start your code after this line
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)
data = {
        'attributes': pd.DataFrame(data.columns[2:10], columns=['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']),
        'target':pd.DataFrame(data.columns[-1], columns=['Glass'])
        }
x_train, x_test, y_train, y_test = train_test_split(dt['attributes'], dt['target'], test_size=0.3, random_state=1)
dt['train'] = {
    'attributes': x_train,
    'target': y_train
  }
dt['test'] = {
  'attributes': x_test,
  'target': y_test
  }
for i in range(3,100,2){ 
    knc = KNeighborsClassifier(i)
    
    input_columns = iris['attributes'].columns[:2].tolist()
    x_train = iris['train']['attributes'][input_columns]
    y_train = iris['train']['target'].Glass
    knc.fit(x_train, y_train)

    x_test = iris['test']['attributes'][input_columns]
    y_test = iris['test']['target'].Glass
    y_predict = knc.predict(x_test)

    print(f'Accuracy: {knc.score(x_test,y_test):.4f}')
    knc += 2
    }
colormap = cm.get_cmap('tab20')
cm_dark = ListedColormap(colormap.colors[::2])
cm_light = ListedColormap(colormap.colors[1::2])

plt.scatter(x_train[input_columns[0]], x_train[input_columns[1]], 
            c=y_train, label='Training data', cmap=cm_dark, 
            edgecolor='black', linewidth=1, s=150)
plt.scatter(x_test[input_columns[0]], x_test[input_columns[1]], 
            c=y_test, marker='*', label='Testing data', cmap=cm_dark, 
            edgecolor='black', linewidth=1, s=150)
plt.xlabel(input_columns[0])
plt.ylabel(input_columns[1])
plt.legend()
