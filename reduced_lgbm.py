import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import Normalizer
from joblib import dump
from scipy.stats import uniform, randint


df = pd.read_csv("pitches.csv")

to_pop = ["Unnamed: 0", "pitch_type", "stand", "p_throws", "inning_topbot", "pitcher_name", "catcher_name", "umpire_name", "batter_name", "game_date"]
for item in to_pop:
    df.pop(item)

# THESE ARE FEATURES THAT ACCOUNTED FOR LESS THAN 1% OF FEATURE IMPORTANCE IN TOTAL AFTER
# FITTING THE FIRST "FINAL" GBM MODEL
to_pop = ["was_3_0","pitch_type_FF","release_speed","inning","was_2_0","release_extension","outs_when_up","pitch_type_CH","was_3_2","pitch_type_SI","pitch_type_FT","pitch_type_FC","bat_score","post_bat_score","pitch_type_CU","was_3_1","pitch_type_FS","was_1_1","was_0_1","was_2_1","was_1_2","was_2_2","was_0_2"]
for item in to_pop:
    df.pop(item)

y = df.pop("description").values
y_lookup, y = np.unique(y, return_inverse = True)
X = df.values

print(df.columns)

model_params = {
    'num_leaves': randint(5, 500),
    'max_depth': [-1, 50, 100, 1000, 10000],
    'learning_rate': uniform(.001, 1),
    'n_estimators': [100, 200, 500, 1000, 10000],
    'min_split_gain': [0, .001, .01, .1],
    'subsample': [.1, .3, .5, .8, 1],
    'importance_type': ['split', 'gain']
}
model = lgb.LGBMClassifier(class_weight='balanced', random_state=123, n_jobs=-1, silent=False)
paramSearchCV = RandomizedSearchCV(estimator=model, param_distributions=model_params, n_iter=500, n_jobs=-1, cv=3)

print("searching for best params")

paramSearchCV.fit(X[1:50000], y[1:50000])
print("Best score: %s" % paramSearchCV.best_score_)
print("Best params: %s" % paramSearchCV.best_params_)

dump(paramSearchCV, filename='reduced_lgbm.randcv.joblib')