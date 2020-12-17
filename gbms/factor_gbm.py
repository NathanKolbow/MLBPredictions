import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as gb_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

X = np.load('nn_X.npy')
y = np.load('nn_y.npy')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,stratify=y,random_state=42)

gbm = gb_classifier(learning_rate=5e-3, max_depth=20, n_estimator=1000)
gbm.fit(X_train, y_train)

print(gbm.score(X_test, y_test))