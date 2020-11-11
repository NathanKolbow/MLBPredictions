import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


if __name__ == '__main__':
    df = pd.read_csv("pitches.csv")
    del df["Unnamed: 0"]
    del df["game_date"]

    to_encode = pd.DataFrame()
    for var in ["pitch_type", "stand", "p_throws", "inning_topbot", "pitcher_name", "catcher_name", "umpire_name", "batter_name"]:
        to_encode[var] = df.pop(var).values

    X = np.zeros((df.shape[0], 2083))
    _X = 0
    le = LabelEncoder()
    for col in to_encode:
        le.fit(to_encode[col].values)
        trans = np_utils.to_categorical(le.transform(to_encode.pop(col)))
        
        size = trans.shape[1]
        X[:, _X:_X+size] = trans
        _X += size
        
    le.fit(df['description'].values)
    y = np_utils.to_categorical(le.transform(df.pop('description')))
        
    for col in df:
        X[:, _X] = df.pop(col).values
        _X += 1
        
    np.save('nn_X.npy', X)
    np.save('nn_y.npy', y)