import pandas as pd
import numpy as np
from tensorflow.keras import models,layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from joblib import load



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
    nn_model.load_weights('nn_686.py.wts')
        
    cnn_model = models.Sequential([
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
    cnn_model.load_weights('cnn_7001.py.wts')
    
    gbm = load('reduced_gbm_no_zone.py.joblib')
    
    
    _, X_test, _, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)
    X_cnn_test = X_test.reshape((X_test.shape[0], 13, 2, 1))
    
    nn_pred = nn_model.predict(X_test)
    cnn_pred = cnn_model.predict(X_cnn_test)
    gbm_pred = gbm.predict_proba(X_test)
    
    
    majority_pred = np.apply_along_axis(np.argmax, 1, (nn_pred+0.2)**5 + (cnn_pred+0.2)**5 + (2*gbm_pred)**3)
    y_test = np.apply_along_axis(np.argmax, 1, y_test)
        
    nn_acc = np.sum(np.apply_along_axis(np.argmax, 1, nn_pred) == y_test) / len(y_test)
    cnn_acc = np.sum(np.apply_along_axis(np.argmax, 1, cnn_pred) == y_test) / len(y_test)
    gbm_acc = np.sum(np.apply_along_axis(np.argmax, 1, gbm_pred) == y_test) / len(y_test)
    major_acc = np.sum(majority_pred == y_test) / len(y_test)
    
    print("nn acc: %s\ncnn acc: %s\ngbm acc: %s\nmajority vote acc: %s" % (nn_acc, cnn_acc, gbm_acc, major_acc))
    
    
    
    
    