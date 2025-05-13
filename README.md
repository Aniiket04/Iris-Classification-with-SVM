# Iris-Classification-with-SVM

## Project Overview 

**Project Title : Iris-Classification-with-SVM**
The goal of this project is to build a machine learning model using Support Vector Machines (SVM) to classify the species of iris flowers based on their morphological features (sepal length, sepal width, petal length, and petal width).

## Objectives
**1. Data Exploration and Preprocessing** :
a. Understand the structure and distribution of the Iris dataset.<br>
b. Perform data visualization to analyze feature separability.

**2. Model Building and Training** :
Train an SVM classifier to distinguish between the three iris species (setosa, versicolor, and virginica).

## Project Structure

### 1. Importing Libraries and Loading the iris dataset
pandas for data manipulation
from sklearn.datasets load the iris dataset directly into our python environment
```python
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
```

### 2. Data processing
**Step-1**
```python
dir(iris)
iris.feature_names
iris.target_names
iris.data
```
When you use the dir() function on the iris object from the sklearn.datasets module, it will display the attributes and methods associated with the iris object.

**Step-2**
```python
len(iris.data)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df
df['target']=iris.target
df
iris.target
df[df.target==2]
```
The len(iris.data) returns the number of rows in the iris.data array, which represents the total number of samples in the dataset.
df = pd.DataFrame(iris.data, columns=iris.feature_names) creates a pandas DataFrame from iris.data, with columns named after iris.feature_names.
df[df.target == 2] filters the DataFrame to include only rows where target equals 2 (virginica species).

**Step-3**
```python
df['flower_names']=df.target.apply(lambda x:iris.target_names[x])
df
df[45:55]
df0=df[:50]
df1=df[50:100]
df2=df[100:150]
```
df['flower_names']=df.target.apply(lambda x:iris.target_names[x]) creates a new column flower_names mapping numerical targets (0, 1, 2) to their corresponding flower species (setosa, versicolor, virginica).
df0=df[:50]
df1=df[50:100]
df2=df[100:150] 
Creates three separate DataFrames:
df0: Rows 0–49, containing only the setosa species.
df1: Rows 50–99, containing only the versicolor species.
df2: Rows 100–149, containing only the virginica species.

### 3. Ploting the graph
Create a scatter plot to visualize the two datasets (df0 and df1) with different colors and markers.
```python
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0["sepal length (cm)"],df0["sepal width (cm)"],color="green",marker="+")
plt.scatter(df1["sepal length (cm)"],df1["sepal width (cm)"],color="red",marker=".")
```

### 4. Train/Test Split
```python
from sklearn.model_selection import train_test_split
x=df.drop(["target","flower_names"],axis="columns")
x
y=df.target
y
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
len(x_train)
```
train_size=0.8 means we intend to split our dataset into 80% training data and 20% testing data

### 5.Model Training
```python
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
```

### 6. Model Prediction
Predictions on the iris dataset.
```python
model.predict([[6,3,5,1.7]])
model.score(x_test,y_test)
```
The model.score() function evaluates the performance of a trained model.

## Conclusion
The objective of this project was to classify the Iris flower species (setosa, versicolor, and virginica) using Support Vector Machines (SVM) and the well-known Iris dataset. The SVM model achieved an accuracy of approximately 96.66% on the test dataset. The features (sepal and petal dimensions) are highly discriminative for the Iris species, especially the petal dimensions, which contributed significantly to the model's performance.

## Author - Aniket Pal
This project is part of my portfolio, showcasing the machine learning skills essential for data science roles.

-**LinkedIn**: [ www.linkedin.com/in/aniket-pal-098690204 ]
-**Email**: [ aniketspal04@gmail.com ]


