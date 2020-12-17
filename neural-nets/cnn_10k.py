import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import keras

import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Source: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/


model = 0

def build_nn_model():

    # 32, 32, 32, 32: 69%
    # 32, 32, 32, 32 w/ conv: 25 7x7, 16 5x5, pool 4x4 s=3, 9 3x3, 9 3x3, pool 2x2 s=2, 9 3x3: 
    nn_model = models.Sequential([
        layers.Conv2D(25, (7, 7), activation='relu', input_shape=(28, 37, 1)),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.Conv2D(9, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=1),
        layers.Conv2D(9, (3, 3), activation='relu'),
        layers.Conv2D(9, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=1),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    nn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    global model
    model = nn_model
    
    return nn_model
    

if __name__ == '__main__':
    X = np.load('nn_X_10k.npy')
    y = np.load('nn_y_10k.npy')
    
    X = X.reshape((X.shape[0], 28, 37, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify=y)
    
    
    estimator = KerasClassifier(build_fn=build_nn_model, epochs=1000, verbose=1,
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=100, restore_best_weights=True)])
    estimator.fit(X_train, y_train)
    
    print(estimator.score(X_test, y_test))
    
    model.save_weights('cnn_10k.py.wts')
    
        