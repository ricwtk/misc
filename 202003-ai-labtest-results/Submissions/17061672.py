#sorry sir I did not have my laptop with me atm, so I could not put my code in .py as I did it through my iPad, apologies for the inconvenience caused.

import pandas as pd
import pyplot from matplotlib

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



def loadDataSet (fullPath):
			
	data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)
	data = data.values
	x,y = data[:,:-1], data[:,-1]
	y = LabelEncoder().fit_transform(y)
	return x,y
	
def evaluate (x,y,model):
	cv = repeat(n_splits = 5, n_repeat =3, random_state = 1)
	scores = cross_Score(model,x,y,scoring="accuracy", cv=cv, n_jobs = -1)
	return scores
	
def get_models():
	models, names = list(), list()
	# KNN
	models.append(KNeighborsClassifier())
	names.append('KNN')
	return models, names
 
full_path = 'glass.csv'
x, y = load_dataset(full_path)
models, names = get_models()
results = list()

for i in range(len(models)):
	scores = evaluate(x, y, models[i])
	results.append(scores)
	# summarize performance
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
	
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()


