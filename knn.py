import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

# Read and split the data into training and testing
X = np.load('nn_X.npy')
y = np.load('nn_y.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)


# Fit the model w/o normalization first; 100 was found to locally maximize prediction accuracy
knn = knn_classifier(n_neighbors=100)
knn.fit(X_train, y_train)

# Evaluate the model; the accuracy is around 0.58 depending on the train/test split
print(knn.score(X_test, y_test))


# Fit the model w/ normalization
norm = Normalizer()
norm.fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)

knn.fit(X_train_norm, y_train)

# Evaluate the model
print(knn.score(X_test_norm, y_test))