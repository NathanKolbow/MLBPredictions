import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import keras
from joblib import dump


import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# Source: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/


class KerasGridOptimizer():
    def __init__(self, epochs=10, n_layers=1, n_nodes=32, learning_rate=5e-4, batch_size=128):
        self._epochs = epochs
        self._n_layers = n_layers
        self._n_nodes = n_nodes
        self._learn_rate = learning_rate
        self._batch_size = batch_size
        
    def get_params(self, deep=False):
        return { 'epochs':self._epochs, 'n_layers':self._n_layers, 'n_nodes':self._n_nodes }
        
    def set_params(self, epochs=None, n_layers=None, n_nodes=None, learning_rate=None, batch_size=None):
        if epochs is not None:
            self._epochs = epochs
        if n_layers is not None:
            self._n_layers = n_layers
        if n_nodes is not None:
            self._n_nodes = n_nodes
        if learning_rate is not None:
            self._learn_rate = learning_rate
        if batch_size is not None:
            self._batch_size = batch_size
        return self
        
    def fit(self, X, y):
        self._model = models.Sequential()
        self._model.add(layers.Flatten(input_shape=(26,)))
        for i in range(self._n_layers):
            self._model.add(layers.Dense(self._n_nodes, activation='relu'))
        self._model.add(layers.Dense(3, activation='softmax'))
            
        opt = keras.optimizers.Adam(learning_rate=self._learn_rate)
        self._model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        self._est = KerasClassifier(build_fn=lambda:self._model, epochs=self._epochs, verbose=0, batch_size=self._batch_size)
        
        self._est.fit(X, y)
        
    def score(self, X, y):
        return self._est.score(X, y)
    
    

if __name__ == '__main__':
    df = pd.read_csv("pitches.csv")

    to_pop = ["Unnamed: 0", "pitch_type", "stand", "p_throws", "inning_topbot", "pitcher_name", "catcher_name", "umpire_name", "batter_name", "game_date"]
    for item in to_pop:
        df.pop(item)

    # THESE ARE FEATURES THAT ACCOUNTED FOR LESS THAN 1% OF FEATURE IMPORTANCE IN TOTAL AFTER
    # FITTING THE FIRST "FINAL" GBM MODEL
    to_pop = ["was_3_0","pitch_type_FF","release_speed","inning","was_2_0","release_extension","outs_when_up","pitch_type_CH","was_3_2","pitch_type_SI","pitch_type_FT","pitch_type_FC","bat_score","post_bat_score","pitch_type_CU","was_3_1","pitch_type_FS","was_1_1","was_0_1","was_2_1","was_1_2","was_2_2","was_0_2","description","zone"]
    for item in to_pop:
        df.pop(item)
    X = df.values
    
    y = np.load('nn_y.npy')
    
    
    model = KerasGridOptimizer()
    param_gr = dict(epochs=[25, 75, 150, 250, 400], 
                    n_layers=[1, 2, 4, 8, 16, 32, 64, 128],
                    n_nodes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                    learning_rate=[0.5, 0.01, 0.001],
                    batch_size=[128, 1024, 8192]
                   )
    grid = GridSearchCV(estimator=model, param_grid=param_gr, verbose=2, n_jobs=-1, cv=3)
    print("Fitting grid.")
    grid.fit(X, y)
    dump(grid, filename='gridcv.joblib')
    
    print("Best score: %s" % grid.best_score_)
    print("Best params: %s" % grid.best_params_)
    