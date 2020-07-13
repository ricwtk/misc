import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from math import fabs, inf
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)


#ignore this part, no time to change
def model(x, m, c):
  return m*x + c

def cost(y, yh):
  return ((y-yh)**2).mean()

def derivatives(x, y, yh):
  return {
    'm': ((y-yh)*x).mean()*-2,
    'c': (y-yh).mean()*-2
  }
  
#to this
  



#assigning the value of the glass.csv into a variable named array
array = data.values

#for attribute
x = array[:,1:8] 
#for target
y = array[:,9] 

#defining the values of test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#creating list for the knc to be stored
list_knc = []


for k in range(3,100): #loop from 3 to 100 index
  knc = KNeighborsClassifier(k)
  knc.fit(x_train, y_train)
  #prediction of line
  y_predict = knc.predict(x_test)
  list_knc.append(knc.score(x_test,y_test))
  print(f'acuracy: {knc.score(x_test,y_test):.4f}')

#plotting of the graphs
plt.figure(figsize=[10,5])
plt.plot(list_knc)
plt.ylabel('acuracy')
plt.xlabel('k')
plt.show()
  