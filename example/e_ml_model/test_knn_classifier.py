import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from example.e_data.fetch_data import load_iris_data

df, target_name = load_iris_data()

# I selected only the petal length and petal width features for classification.
X = df[['petal length', 'petal width']]
y = df[target_name]

print(X.head())
print(y.head())