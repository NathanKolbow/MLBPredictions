import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline


# Source: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/


if __name__ == '__main__':
    X = np.load('nn_X.npy')
    y = np.load('nn_y.npy')
    
    nn_model = models.Sequential([
        layers.Flatten(input_shape=(2083,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    estimator = KerasClassifier(build_fn=nn_model, epochs=10, batch_size=1000, verbose=1)
    estimator.fit(X_train, y_train)
    print(estimator.score(X_test, y_test))
    
        