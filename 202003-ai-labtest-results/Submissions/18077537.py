import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"],
                   index_col=0)

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
    'target': pd.DataFrame(data, columns=['Glass type']),
    'targetNames': list(data.columns)
}

X_train, X_test, Y_train, Y_test = train_test_split(glass['attributes'], glass['target'], test_size=0.30, random_state = 1)

glass['train'] = {
    'attributes': X_train,
    'target': Y_train
}
glass['test'] = {
    'attributes': X_test,
    'target': Y_test
}

kArr = []
accuracy = []

for i in range(3, 101):
    if i % 2 == 1:
        knn = KNeighborsClassifier(i)

        col = glass['attributes'].columns.tolist()

        X_train = glass['train']['attributes'][col]
        Y_train = glass['train']['target'].values.ravel()
        knn.fit(X_train, Y_train)

        X_test = glass['test']['attributes'][col]
        Y_test = glass['test']['target'].values.ravel()
        Y_pred = knn.predict(X_test)

        # Output
        print(f'k = {i}')
        print(pd.DataFrame(list(zip(Y_test, Y_pred)), columns=['target', 'predicted']))
        print(f'Accuracy: {knn.score(X_test, Y_test):.4f}')

        kArr.append(i)
        accuracy.append(knn.score(X_test, Y_test))

plt.figure(figsize=[15, 10])
plt.title('K vs Accuracy')

plt.ylim(ymin=0)

plt.scatter(kArr, accuracy)
plt.xlabel('K')
plt.ylabel('Accuracy')

# plt.waitforbuttonpress()
