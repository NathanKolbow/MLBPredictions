import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
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

X_train_temp, X_test, y_train_temp, y_test = train_test_split(X,y,test_size=.8,stratify=y,random_state=123)
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=.3,stratify=y,random_state=1415)

# model_params = {
#     'objective': ['multiclass', 'multiclassova'],
#     'tree_learner': ['serial', 'data', 'voting']
# }

model = lgb.LGBMClassifier(num_class=3, class_weight='balanced', importance_type='gain', min_split_gain=.25, num_leaves=100, learning_rate=.01, subsample=.9, subsample_freq=1, feature_fraction=.9, lambda_l1=1.545, lambda_l2=.215, n_iter_no_change=60, num_iterations=10000, max_depth=2000, random_state=42, n_jobs=-1, silent=True)
# model = lgb.LGBMClassifier(device_type='gpu', gpu_use_dp=True, num_class=3, boosting_type='gbdt', class_weight='balanced', importance_type='gain', subsample=.9, subsample_freq=1, feature_fraction=.9, random_state=42, n_jobs=1, silent=True)

# # paramSearchCV = RandomizedSearchCV(estimator=model, param_distributions=model_params, n_jobs=-1, cv=3)
# paramSearchCV = GridSearchCV(estimator=model, param_grid=model_params, n_jobs=-1, cv=3)


# print("Searching for best params (6? iter) ...")

# paramSearchCV.fit(X_train[1:20000], y_train[1:20000], eval_set=(X_valid, y_valid))
# print("Best score: %s" % paramSearchCV.best_score_)
# print("Best params: %s" % paramSearchCV.best_params_)

# best_model = model.set_params(**paramSearchCV.best_params_)
best_model = model

# print("Fitting best model (all records) ... ")
# best_model.fit(X_train_temp, y_train_temp, eval_set=(X_valid, y_valid))

# print("Best model score: %s" % best_model.score(X_test, y_test))
# print("Best model params: %s" % best_model.get_params)




bootstrap_train_accuracies = []

idx = np.arange(y_train_temp.shape[0])
rng = np.random.RandomState(seed=12345)
start_bootstrap = time.perf_counter()
for i in range(100):
    train_idx = rng.choice(idx, size=idx.shape[0], replace=True)
    test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
    
    boot_train_X, boot_train_y = X_train_temp[train_idx], y_train_temp[train_idx]
    boot_test_X, boot_test_y = X_train_temp[test_idx], y_train_temp[test_idx]
    
    best_model.fit(boot_train_X, boot_train_y, eval_set=(X_valid, y_valid))

    bootstrap_train_accuracies.append(best_model.score(boot_test_X, boot_test_y))
end_bootstrap = time.perf_counter()
print(f"100 rounds bootstrap optimized LGBM took {end_bootstrap - start_bootstrap:0.4f} seconds")
print(f"Avg: {(end_bootstrap - start_bootstrap)/100.0:0.4f} seconds")


bootstrap_percentile_lower = np.percentile(bootstrap_train_accuracies, 2.5)
bootstrap_percentile_upper = np.percentile(bootstrap_train_accuracies, 97.5)
print("95% Bootstrap CI:")
print(bootstrap_percentile_lower, bootstrap_percentile_upper)
print("Mean: ", bootstrap_train_mean)


fig, ax = plt.subplots(figsize=(8, 4))
ax.vlines( bootstrap_train_mean, [0], 80, lw=2.5, linestyle='-', label='bootstrap train mean')
ax.vlines(bootstrap_percentile_upper, [0], 15, lw=2.5, linestyle='dashed', 
          label='CI95 bootstrap, percentile', color='C1')
ax.vlines(bootstrap_percentile_lower, [0], 15, lw=2.5, linestyle='dashed', color='C1')

ax.hist(bootstrap_train_accuracies, bins=7,
        color='#0080ff', edgecolor="none", 
        alpha=0.3)
plt.legend(loc='upper left')
plt.xlim([0.69, 0.75])
plt.tight_layout()
plt.savefig('figures/lgbm-tuned-bootstrap-ci-histo.svg')
# plt.show()
plt.clf()


# dump(paramSearchCV, filename='reduced_lgbm.randcv.joblib')
# dump(best_model, filename='reduced_lgbm.bestlgbm.joblib')