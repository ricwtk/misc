import pandas as pd
# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)


'''(a) split the data into 70% training and 30% testing data
'''
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

array = data.values
x = array[:,1:8] # attribute
y = array[:,9] # target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

'''
(b) plot the accuracy of knn classifiers for all odd value of k between 3 to 100, i.e. k = 3, 5, 7, : : :, 100.
'''
kncList = []

for k in range(3,101):
  knc = KNeighborsClassifier(k)
  knc.fit(x_train, y_train)
  y_predict = knc.predict(x_test)
  kncList.append(knc.score(x_test,y_test))
  print("For K-score = ", k)
  print(f'Accuracy: {knc.score(x_test,y_test):.4f}')
  
'''visualisation
'''

plt.figure(figsize=[12,8])
plt.plot(kncList)
plt.ylabel('accuracy')
plt.xlabel('k-value')
plt.show()