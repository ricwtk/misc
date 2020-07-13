import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor



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

data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)
import glass.csv as DataFrame
data = {'Glass': pd.DataFrame(data.data, columns=data.names), 
        'target':pd.DataFrame(data.target, columns=['glass_type'])
        }

from sklearn.model_selection import train_test_split
for names in [data]: 
    x_train, x_test = train_test_split(data['Glass'], data['target'] ,test_size=0.3, random_state=1)
    data['train'] = {
        'Glass': x_train,
        'target': x_test
        }
    
knc = KNeighborsClassifier(3)
input_columns = data['Glass'].columns[:2].tolist()
x_train = data['Glass']['target'][input_columns]
knc.fit(x_train)

x_test = data['test']['attributes'][input_columns]
x_predict = knc.predict(x_test)

print(pd.DataFrame(list(zip(x_test,x_predict)), columns=['target', 'predicted']))
print(f'Accuracy: {knc.score(x_test):.4f}')

'''
I never learnt python, so i never understood to work with python
It's not that i had time to practise and learn python
I had to dedicate all my time to Capstone 2
I tried my best... 
but im probably going to get a zero for this aren't I? :( 

'''
