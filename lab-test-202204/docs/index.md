---
hide:
  - navigation
---
# Lab test (April 2022 Semester)

This lab test is an individual test contributing 10% to the total grade of CSC3206 Artificial Intelligence.

## Instructions

1. The duration of this lab test is 1 hour.
2. You are given a data file (`dataset.csv`) and an incomplete Python script file (`script.py`).
3. Complete the tasks specified in the next section and submit the `script.py` through the submission link provided.
4. In `script.py` file, you should only add or change codes that are enclosed within the blocks of

    ````
    ########## student's code ##########
    # your task description

    ####################################
    ````

5. The codes within the blocks will be extracted automatically for marking purpose, therefore, do not add or modify code outside of these blocks as it will not be there during marking.
6. You may run the `script.py` file to test your codes. It should run without error.

## Submission

Only submit the `script.py` file.

You do not need to

1. rename the file
2. zip the file
3. submit the csv files

## Explanation of tasks

The `script.py` is meant to 

1. perform clustering on data loaded from the `dataset.csv` file
2. perform classification on the data with the clustering result
3. save the outcome to a `csv` file

Each task corresponds to one custom function. The data flow from one task/function to the next one has been done in the `if __name__ == "__main__"` block at the end of `script.py`. You only need to tackle the tasks in each function.

```python
df = load_file()
kmModel = train_clustering_model(df)
results = test_clustering_model(df, kmModel)
df = add_clustering_result_to_data(df, kmModel)

dtModel = train_decision_tree(df)
df = test_decision_tree(df, dtModel)
save_to_file(df)
```

### Task 1
Complete your name and student ID.

```python
def student_details():
  ########## student's code ##########
  ##########    block 1     ##########
  # 1. Update the name and id to your name and student id
  studentName = ""
  studentId = ""
  ####################################
  return studentName, studentId
```

### Task 2
1. Load the `datasest.csv` file as a pandas DataFrame with variable name `df` (take note that the csv file has no header).
2. Add/set the headers of the columns (from left to right) to be
    
    ```
    area, perimeter, compactness, length, width, asymmetry, groove
    ```

```python
def load_file():
  ########## student's code ##########
  ##########    task 2     ##########
  # 1. load the dataset.csv file to be a pandas DataFrame with vairable name: "df"
  #    (note that the csv file has no header)
  #    add/set the headers of the columns (from left to right) to be
  #        area, perimeter, compactness, length, width, asymmetry, groove
  
  ####################################
  return df
```

### Task 3
1. Initialise a kmeans model with 5 clusters using variable name `kmModel`.
2. Train `kmModel` using any three columns from `df`.
    1. `df` is the DataFrame created in Task 2.
    2. You may choose and hardcode any three columns of the `df`.

```python
def train_clustering_model(df):
  ########## student's code ##########
  ##########    task 3     ##########  
  # 1. initialise a kmeans model with 5 clusters using variable name: "kmModel"
  # 2. train the kmeans model using any three columns from the df
  
  ####################################
  return kmModel
```

### Task 4
1. Identify/predict the clusters of any 10 rows from `df`.
    1. `df` is the DataFrame used in Task 3.
    2. `kmModel` is the trained kmeans model from Task 3.
2. Save the identified cluster index (from 1) with variable name `outcome`.

```python
def test_clustering_model(df, kmModel):
  ########## student's code ##########
  ##########    task 4     ##########  
  # 1. use any 10 rows from df and identify/predict their clusters
  # 2. save the identified cluster index with variable name: "outcome"
  
  ####################################
  return outcome
```

### Task 5
1. Predict the clusters of every row in `df`.
    1. `df` is the DataFrame used in Task 4.
    2. `kmModel` is the trained kmeans model from Task 3.
2. Convert the predicted cluster numbers (`0`,`1`,`2`,`3`,`4`) to alphabets (`a`,`b`,`c`,`d`,`e`).
3. Add the outcome as a new column of `df` with the name `cresult`.

```python
def add_clustering_result_to_data(df, kmModel):
  ########## student's code ##########
  ##########    task 5     ##########  
  # 1. predict the clusters of every row in df
  # 2. convert the cluster numbers (0,1,2,3,4) to alphabets (a,b,c,d,e)
  # 3. add the cluster outcome as a new column called "cresult"
  
  ####################################
  return df
```

### Task 6
1. Initialise a decisison tree model with maximum depth of 5 using variable name `dtModel`.
2. Train the decision tree model to classify using the inputs of `length`, `width`, and `groove` to identify the output of `cresult`.
    1. `df` is the DataFrame from Task 5.

```python
def train_decision_tree(df):
  ########## student's code ##########
  ##########    task 6     ##########  
  # 1. initialise a decision tree model with maximum depth of 5 using variable name: dtModel
  # 2. train the decision tree model to classify based on inputs of
  #    a. length
  #    b. width
  #    c. groove
  #    to identify the output of "cresult"
  
  ####################################
  return dtModel
```

### Task 7
1. Predict the class of all the data in `df` using the trained decision tree `dtModel`.
    1. `df` is the DataFrame used in Task 6.
    2. `dtModel` is the decision tree model created in Task 6.
2. Add the predicted outcome as a new column of `df` with the name `dresult`.

```python
def test_decision_tree(df, dtModel):
  ########## student's code ##########
  ##########    task 7     ##########  
  # 1. predict the class using the trained decision tree
  # 2. add the predicted outcome as a new column of df called "dresult"
  
  ####################################
  return df
```

### Task 8
1. Save the DataFrame `df` to a csv file with the name of `finalresults.csv`.
    1. `df` is the DataFrame from Task 7.

```python
def save_to_file(df):
  ########## student's code ##########
  ##########    task 8     ##########  
  # 1. save the dataframe "df" to a csv file with the name of "finalresults.csv"
  
  ####################################
```

## Submission

Only submit the `script.py` file.

You do not need to

1. rename the file
2. zip the file
3. submit the csv files