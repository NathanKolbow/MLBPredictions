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
    # 32, 32, 32, 32: ~67% acc
    # 24, 24, 24, 24: ~66.5% acc
    # 64, 64, 64, 64: ~67% acc
    # 64, 64, 64, 64, 64, 64, 64, 64: ~36% acc
    # 4x32: ~36% acc
    # 4x16: ~36% acc
    #
    # Added pitcher_name to the onehot data
    # 32, 32, 32, 32: ~67%
    # train=80%, test=10%, valid=10%
    # 32, 32, 32, 32: ~64%
    # train=50%, test=10%, valid=40%
    # 32, 32, 32, 32: ~58%
    # train=70%, test=15%, valid=15%
    # 512, 128, 32, 32: 67%
    # Now using only the 27 variables from GBM feature selection
    # 64, 64: 67.7%
    # train=80%, test=20%, no valid, only early-stopping on loss
    # 64, 64: 67.4%
    # 32, 64, 32: ~67%
    # 64, 64, 64, 64: 68.2%
    # 64, 64, 64, 64, 64: 61%
    # 32, 64, 64, 64, 32: 67%
    # 32, 64, 32, 64, 32: 
    # 'zone' removed
    # 64, 64, 64, 64: 67%
    # 64, 64, 64, 64, 64: 68.6%
    nn_model = models.Sequential([
        layers.Flatten(input_shape=(26,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    opt = keras.optimizers.Adam(learning_rate=0.005)
    nn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42, shuffle=True)
    
    estimator = KerasClassifier(build_fn=build_nn_model, epochs=5000, verbose=1, batch_size=128,
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)])
    estimator.fit(X_train, y_train)
    
    print(estimator.score(X_test, y_test))

    model.save_weights('nn.py.wts')
    
        