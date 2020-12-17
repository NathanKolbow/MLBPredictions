import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as gb_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from joblib import dump


df = pd.read_csv("pitches.csv")

to_pop = ["Unnamed: 0", "pitch_type", "stand", "p_throws", "inning_topbot", "pitcher_name", "catcher_name", "umpire_name", "batter_name", "game_date"]
for item in to_pop:
    df.pop(item)

# THESE ARE FEATURES THAT ACCOUNTED FOR LESS THAN 1% OF FEATURE IMPORTANCE IN TOTAL AFTER
# FITTING THE FIRST "FINAL" GBM MODEL
to_pop = ["was_3_0","pitch_type_FF","release_speed","inning","was_2_0","release_extension","outs_when_up","pitch_type_CH","was_3_2","pitch_type_SI","pitch_type_FT","pitch_type_FC","bat_score","post_bat_score","pitch_type_CU","was_3_1","pitch_type_FS","was_1_1","was_0_1","was_2_1","was_1_2","was_2_2","was_0_2", "plate_x", "plate_z"]
for item in to_pop:
    df.pop(item)

y = df.pop("description").values
y_lookup, y = np.unique(y, return_inverse = True)
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,stratify=y,random_state=42)

gbm = gb_classifier(subsample=0.7, learning_rate=0.1, max_depth=5, n_estimators=300, verbose=1)
gbm.fit(X_train, y_train)

print(gbm.score(X_test, y_test))
print(gbm.feature_importances_)
dump(gbm, 'reduced_gbm.py.joblib')