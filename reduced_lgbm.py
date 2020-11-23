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

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,stratify=y,random_state=123)

model_params = {
    'num_leaves': randint(5, 500),
    'max_depth': [-1, 50, 100, 1000, 2000],
    'learning_rate': uniform(.001, 1),
    'num_iterations': [200, 500, 1000, 10000, 20000],
    'min_split_gain': [0, .001, .01, .1, .2, .3],
    'subsample': [.1, .3, .5, .8, 1],
    'n_iter_no_change':[15, 30, 45, 60, 100],
    'importance_type': ['split', 'gain']
}
model = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, silent=True)
paramSearchCV = RandomizedSearchCV(estimator=model, param_distributions=model_params, n_iter=500, n_jobs=-1, cv=3)

print("Searching for best params (500 iter) ...")

paramSearchCV.fit(X_train[1:200000], y_train[1:200000], eval_set=(X_test, y_test))
print("Best score: %s" % paramSearchCV.best_score_)
print("Best params: %s" % paramSearchCV.best_params_)

best_model = model.set_params(**paramSearchCV.best_params_)

print("Fitting best model (all records) ... ")
best_model.fit(X_train, y_train, eval_set=(X_test, y_test))

print("Best model score: %s" % best_model.score(X_test, y_test))
print("Best model params: %s" % best_model.get_params)

dump(paramSearchCV, filename='reduced_lgbm.randcv.joblib')
dump(best_model, filename='reduced_lgbm.bestlgbm.joblib')