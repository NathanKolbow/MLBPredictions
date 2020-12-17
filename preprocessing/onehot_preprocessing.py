import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


if __name__ == '__main__':
    df = pd.read_csv("pitches.csv")
    del df["Unnamed: 0"]
    del df["game_date"]
    del df['on_3b_yes_no']
    del df['on_2b_yes_no']
    del df['on_1b_yes_no']
    del df['pitch_type_CH']
    del df['pitch_type_CU']
    del df['pitch_type_FC']
    del df['pitch_type_FF']
    del df['pitch_type_FS']
    del df['pitch_type_FT']
    del df['pitch_type_SI']
    del df['pitch_type_SL']
    del df['umpire_name']
    del df['pitcher_name']
    del df['catcher_name']
    del df['batter_name']
    del df['stand']
    del df['inning_topbot']

    df['p_throws'] = [1 if x == 'R' else 0 for x in df['p_throws']]

    to_encode = pd.DataFrame()
    for var in ["pitch_type", "outs_when_up"]:
        to_encode[var] = df.pop(var).values
    df['points_on_pitch'] = df['post_bat_score'] - df['bat_score']
    del df['post_bat_score']
    del df['bat_score']

    X = np.zeros((df.shape[0], 49))
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
    
    np.save('nn_X_10k.npy', X[0:50000, :])
    np.save('nn_y_10k.npy', y[0:50000, :])