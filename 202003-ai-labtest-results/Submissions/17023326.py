import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)

dt = pd.DataFrame(data, columns=["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"])
y = pd.DataFrame(data, columns=['Glass type'])
x_train, x_test, y_train, y_test = train_test_split(dt, y, test_size=0.3, random_state=1)

K = list(range(3, 100, 2))
ACC = [] 
for k in K:
     knn = KNeighborsClassifier(k)
     knn.fit(x_train, np.ravel(y_train))
     y_predict = knn.predict(x_test)
     acc = knn.score(x_test, y_test)
     ACC.append(acc)
     print(f'When k is {k}, Accuracy: {acc:.4f}')
plt.figure()
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.plot(K, ACC, 'go-',label='line 1', linewidth=2)

