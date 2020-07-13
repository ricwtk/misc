import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn  import  metrics
import matplotlib.pyplot as pt
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

# start your code aftefr this line
accuracy_list_plot=[]
x= data.drop(columns=["Glass type"])
y=data["Glass type"].values
input_columns=['Na','Mg','Al','Si','K','Ca','Ba','Fe']  
list_of_k=[3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99]
for i in list_of_k:
    knc=KNeighborsClassifier(i)
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3)
    knc.fit(x_train,y_train)
    predictions = knc.predict(x_test)
    accuracy= metrics.accuracy_score(y_test,predictions)
    accuracy_list_plot.append(accuracy)
    print("accuracy",accuracy)  

pt.xlabel("Value K")
pt.ylabel("Accuracy")
pt.plot(list_of_k,accuracy_list_plot)
pt.show()


