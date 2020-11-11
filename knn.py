import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

# Read and split the data into training and testing
df = pd.read_csv("pitches.csv")
del df['Unnamed: 0']
factor_vars = ["game_date", "pitch_type", "stand", "p_throws", "inning_topbot", "pitcher_name", "catcher_name", "umpire_name", "batter_name"]
for var in factor_vars:
    del df[var]

y = df.pop('description').values
X = df.values
del df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)


for i in [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
    # Fit the model w/o normalization first; 100 was found to locally maximize prediction accuracy
    knn = knn_classifier(n_neighbors=i)
    knn.fit(X_train, y_train)

    # Evaluate the model; the accuracy is around 0.58 depending on the train/test split
    print("k=%s: %s" % (i, knn.score(X_test, y_test)))
"""
    Results:
        k=5: 0.5456448100081732
        k=10: 0.5619062017225169
        k=15: 0.5671266659422584
        k=25: 0.5731786249155513
        k=50: 0.5781534202308343
        k=75: 0.5777849168741467
        k=100: 0.577288854663221
        k=125: 0.5770242881507274
        k=150: 0.5757581484123647
        k=175: 0.5749030316487691
        k=200: 0.5745439770960991
        k=225: 0.5736888603325034
        k=250: 0.5726778383026169
        k=275: 0.5718132727350036
        k=300: 0.5701455588258916
"""


# Fit the model w/ normalization
norm = Normalizer()
norm.fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)

for i in [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
    knn = knn_classifier(n_neighbors=i)
    knn.fit(X_train_norm, y_train)
    # Evaluate the model
    print("k=%s: %s" % (i, knn.score(X_test_norm, y_test)))