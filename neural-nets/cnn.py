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
    # 32, 32, 32, 32: ~68% acc
    # 16, 16, 16, 16: ~65% acc
    # 64, 64: ~67% acc
    # 128, 128: ~67% acc
    # 64, 64, 64, 64: ~67% acc
    #
    # Changed vars to only the 27 vars selected from GBM feature importance
    # train=80%, test=20%, no valid, just early stopping
    # 64, 64, 64, 64: 69.7%
    # 64, 64, 64, 64, 64: 69.9%
    # 64, 64, 64, 64, 64, 64: 70.01%
    nn_model = models.Sequential([
        layers.Conv2D(9, (2, 2), activation='relu', input_shape=(13, 2, 1)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    nn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    global model
    model = nn_model
    
    return nn_model
    

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
    
    X = X.reshape((X.shape[0], 13, 2, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)
    
    estimator = KerasClassifier(build_fn=build_nn_model, epochs=5000, verbose=1, batch_size=128,
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)])
    estimator.fit(X_train, y_train)
    
    print(estimator.score(X_test, y_test))
    
    model.save_weights('cnn.py.wts')
    
        