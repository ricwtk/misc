import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

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

dt = {
  'attributes': pd.DataFrame(data, columns=["RI","Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]),
  'target': pd.DataFrame(data, columns=['Glass type']),
  'targetNames': ['Glass type']
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
print(dt['train'])

knnC = []
accuracy = []

for i in range(3,100,2):
    knc = KNeighborsClassifier(i)
    
    knnC.append(i)
    input_columns = dt['attributes'].columns.tolist()
    target_columns = dt['target'].columns.tolist()
    x_train = dt['train']['attributes']
    y_train = dt['train']['target']
    knc.fit(x_train, y_train)
    
    x_test = dt['test']['attributes']
    y_test = dt['test']['target']
    y_predict = knc.predict(x_test)
    
    acc = knc.score(x_test,y_test)
    accuracy.append(acc)


    print()
    print(pd.DataFrame(list(zip(y_test,y_predict)), columns=['target', 'predicted']))
    print(f'Accuracy for k = {i}: {acc:.4f}')
    

plt.figure(figsize=[12,8])

plt.plot(knnC,accuracy)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('Optimal k')

plt.show()