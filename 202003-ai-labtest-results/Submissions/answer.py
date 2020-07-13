import pandas as pd

data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]], data["Glass type"], test_size=0.3)

print(x_train.shape, x_test.shape)

from sklearn.neighbors import KNeighborsClassifier

k_list = [k for k in range(3,100,2)]
accuracy = []
for k in k_list:
  knc = KNeighborsClassifier(k)
  knc.fit(x_train, y_train)

  y_predict = knc.predict(x_test)
  accuracy.append(knc.score(x_test,y_test))
  print("k = {}, accuracy = {}".format(k, accuracy[-1]))

import matplotlib.pyplot as plt
plt.plot(k_list, accuracy)

# plt.savefig("test")
