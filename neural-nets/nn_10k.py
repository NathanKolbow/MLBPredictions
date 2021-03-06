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


model = None

def build_nn_model():
    global model
    nn_model = models.Sequential([
        layers.Flatten(input_shape=(1036,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    nn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model = nn_model
    return nn_model
    

if __name__ == '__main__':
    X = np.load('nn_X_10k.npy')
    y = np.load('nn_y_10k.npy')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify=y)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)
    
    
    estimator = KerasClassifier(build_fn=build_nn_model, epochs=1000, verbose=1, validation_data=(X_valid, y_valid),
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)])
    estimator.fit(X_train, y_train)
    
    print(estimator.score(X_test, y_test))

    model.save_weights('nn_10k.py.wts')
    
        