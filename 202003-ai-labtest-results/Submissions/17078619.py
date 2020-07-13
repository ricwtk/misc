import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)

dt = data[['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y = data['Glass type']
X_train,X_test,y_train,y_test = train_test_split(dt,y,test_size = 0.3, random_state=1)

# start your code after this line
k = 3
k_value = []
accuracy = []

while (k <= 100):
    knc = KNeighborsClassifier(k)
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)
    k_value.append(k)
    accuracy.append(knc.score(X_test,y_test))
    print("k: " + str(k) + " " + "acc: " + str(knc.score(X_test,y_test)))
    k += 2
    
plt.title("K-value vs. Accuracy")
plt.plot(k_value,accuracy)
plt.xlabel = ('K-value')
plt.ylabel = ('Accuracy')