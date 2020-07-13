import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)
y=data.Mg
x=data.drop('Mg',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train.head()
x_train.shape
x_test.head()
x_test.shape

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_test, y_test)
y_pred = knn.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_test, y_test)
y_pred = knn.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_test, y_test)
y_pred = knn.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

plt.figure()
plt.pcolormesh(x, y, cmap=dia_cm)

plt.scatter(x_train[input_columns[0]], x_train[input_columns[1]], 
            c=y_train, label='Training data', cmap=dia_cm, 
            edgecolor='black', linewidth=1, s=150)
plt.scatter(x_test[input_columns[0]], x_test[input_columns[1]], 
            c=y_test, marker='*', label='Testing data', cmap=dia_cm,
            edgecolor='black', linewidth=1, s=150)
plt.xlabel(input_columns[0])
plt.ylabel(input_columns[1])
plt.legend()
plt.colorbar()
plt.show()