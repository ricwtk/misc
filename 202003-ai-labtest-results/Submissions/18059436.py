import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
data=pd.DataFrame(data)

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

# Part A
input_features=['Na','Mg','Al','Si','K','Ca','Ba','Fe']
X_train, X_test, y_train, y_test = train_test_split(data[input_features], data["Glass type"], test_size=0.3, random_state=1)

# Part B
kList=[]
accuracyList=[]
for k in range(3,101):
   if (k%2) !=0:
       kList.append(k)
       knc=KNeighborsClassifier(k)
       knc.fit(X_train,y_train)
       accu=knc.score(X_test,y_test)
       print(f'Accuracy: {knc.score(X_test,y_test):.4f}')
       accuracyList.append(accu)

plt.figure()
plt.scatter(kList,accuracyList)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Comparison of Accuracy")
plt.show()
plt.legend()
        
        
        
    
    
