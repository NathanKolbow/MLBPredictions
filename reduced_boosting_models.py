import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.ensemble import GradientBoostingClassifier as gb_classifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from joblib import dump
from scipy.stats import uniform, randint
from mlxtend.evaluate import bootstrap_point632_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score



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

# X_train_temp, X_test, y_train_temp, y_test = train_test_split(X,y,test_size=.2,stratify=y,random_state=123)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train_temp,y_train_temp,test_size=.3,stratify=y,random_state=1415)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,stratify=y,random_state=123) # only need the validation set if we are parameter tuning

gbm = gb_classifier(subsample=0.9, verbose=1)
lgbm = lgb.LGBMClassifier(num_class=3, boosting_type='gbdt', class_weight='balanced', importance_type='gain', min_split_gain=.25, subsample=.9, subsample_freq=1, feature_fraction=.9, random_state=42, n_jobs=-1, silent=True)
dart = lgb.LGBMClassifier(num_class=3, boosting_type='dart', class_weight='balanced', importance_type='gain', min_split_gain=.25, subsample=.9, subsample_freq=1, feature_fraction=.9, random_state=42, n_jobs=-1, silent=True)
goss = lgb.LGBMClassifier(num_class=3, boosting_type='goss', class_weight='balanced', importance_type='gain', min_split_gain=.25, random_state=42, n_jobs=-1, silent=True)


# Need to do this because normal gbm takes too long in the for loop
start_gbm_cv = time.perf_counter()
gbm_cv_acc = cross_val_score(estimator=gbm,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=-1) # very necessary
end_gbm_cv = time.perf_counter()
print('GBM 10-fold Accuracy: %.2f%%' % (np.mean(gbm_cv_acc)*100))
print(f"10-fold CV on GBM with 100000 examples took {(end_gbm_cv - start_gbm_cv):0.4f} seconds")


lgbm_bootstrap_train_accuracies = []
dart_bootstrap_train_accuracies = []
goss_bootstrap_train_accuracies = []

idx = np.arange(y_train.shape[0])
rng = np.random.RandomState(seed=12345)
start_lgbm_bootstrap = time.perf_counter()
for i in range(100):
    train_idx = rng.choice(idx, size=idx.shape[0], replace=True)
    test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
    
    boot_train_X, boot_train_y = X_train[train_idx], y_train[train_idx]
    boot_test_X, boot_test_y = X_train[test_idx], y_train[test_idx]
    
    lgbm.fit(boot_train_X, boot_train_y)

    lgbm_bootstrap_train_accuracies.append(lgbm.score(boot_test_X, boot_test_y))
end_lgbm_bootstrap = time.perf_counter()
print(f"100 rounds bootstrap LGBM took {end_lgbm_bootstrap - start_lgbm_bootstrap:0.4f} seconds")
print(f"Avg: {(end_lgbm_bootstrap - start_lgbm_bootstrap)/100.0:0.4f} seconds")


idx = np.arange(y_train.shape[0])
rng = np.random.RandomState(seed=12345)
start_dart_bootstrap = time.perf_counter()
for i in range(100):
    train_idx = rng.choice(idx, size=idx.shape[0], replace=True)
    test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
    
    boot_train_X, boot_train_y = X_train[train_idx], y_train[train_idx]
    boot_test_X, boot_test_y = X_train[test_idx], y_train[test_idx]
    
    dart.fit(boot_train_X, boot_train_y)

    dart_bootstrap_train_accuracies.append(dart.score(boot_test_X, boot_test_y))
end_dart_bootstrap = time.perf_counter()
print(f"100 rounds bootstrap DART took {end_dart_bootstrap - start_dart_bootstrap:0.4f} seconds")
print(f"Avg: {(end_dart_bootstrap - start_dart_bootstrap)/100.0:0.4f} seconds")


idx = np.arange(y_train.shape[0])
rng = np.random.RandomState(seed=12345)
start_goss_bootstrap = time.perf_counter()
for i in range(100):
    train_idx = rng.choice(idx, size=idx.shape[0], replace=True)
    test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
    
    boot_train_X, boot_train_y = X_train[train_idx], y_train[train_idx]
    boot_test_X, boot_test_y = X_train[test_idx], y_train[test_idx]
    
    goss.fit(boot_train_X, boot_train_y)

    goss_bootstrap_train_accuracies.append(goss.score(boot_test_X, boot_test_y))
end_goss_bootstrap = time.perf_counter()
print(f"100 rounds bootstrap GOSS took {end_goss_bootstrap - start_goss_bootstrap:0.4f} seconds")
print(f"Avg: {(end_goss_bootstrap - start_goss_bootstrap)/100.0:0.4f} seconds")


gbm_cv_acc_mean = np.mean(gbm_cv_acc)
lgbm_bootstrap_train_mean = np.mean(lgbm_bootstrap_train_accuracies)
dart_bootstrap_train_mean = np.mean(dart_bootstrap_train_accuracies)
goss_bootstrap_train_mean = np.mean(goss_bootstrap_train_accuracies)


gbm_bootstrap_percentile_lower = np.percentile(gbm_cv_acc, 2.5)
gbm_bootstrap_percentile_upper = np.percentile(gbm_cv_acc, 97.5)
print("GBM 95% CI (10-fold CV):")
print(gbm_bootstrap_percentile_lower, gbm_bootstrap_percentile_upper)
print("Mean: ", gbm_cv_acc_mean)

lgbm_bootstrap_percentile_lower = np.percentile(lgbm_bootstrap_train_accuracies, 2.5)
lgbm_bootstrap_percentile_upper = np.percentile(lgbm_bootstrap_train_accuracies, 97.5)
print("LGBM 95% Bootstrap CI:")
print(lgbm_bootstrap_percentile_lower, lgbm_bootstrap_percentile_upper)
print("Mean: ", lgbm_bootstrap_train_mean)

dart_bootstrap_percentile_lower = np.percentile(dart_bootstrap_train_accuracies, 2.5)
dart_bootstrap_percentile_upper = np.percentile(dart_bootstrap_train_accuracies, 97.5)
print("DART 95% Bootstrap CI:")
print(dart_bootstrap_percentile_lower, dart_bootstrap_percentile_upper)
print("Mean: ", dart_bootstrap_train_mean)

goss_bootstrap_percentile_lower = np.percentile(goss_bootstrap_train_accuracies, 2.5)
goss_bootstrap_percentile_upper = np.percentile(goss_bootstrap_train_accuracies, 97.5)
print("GOSS 95% Bootstrap CI:")
print(goss_bootstrap_percentile_lower, goss_bootstrap_percentile_upper)
print("Mean: ", goss_bootstrap_train_mean)



fig, ax = plt.subplots(figsize=(8, 4))
ax.vlines( lgbm_bootstrap_train_mean, [0], 80, lw=2.5, linestyle='-', label='lgbm bootstrap train mean')
ax.vlines(lgbm_bootstrap_percentile_upper, [0], 15, lw=2.5, linestyle='dashed', 
          label='CI95 bootstrap, percentile', color='C1')
ax.vlines(lgbm_bootstrap_percentile_lower, [0], 15, lw=2.5, linestyle='dashed', color='C1')

ax.hist(lgbm_bootstrap_train_accuracies, bins=7,
        color='#0080ff', edgecolor="none", 
        alpha=0.3)
plt.legend(loc='upper left')
plt.xlim([0.69, 0.75])
plt.tight_layout()
plt.savefig('figures/lgbm-bootstrap-ci-histo.svg')
# plt.show()
plt.clf()

fig, ax = plt.subplots(figsize=(8, 4))
ax.vlines( dart_bootstrap_train_mean, [0], 80, lw=2.5, linestyle='-', label='dart bootstrap train mean')
ax.vlines(dart_bootstrap_percentile_upper, [0], 15, lw=2.5, linestyle='dashed', 
          label='CI95 bootstrap, percentile', color='C1')
ax.vlines(dart_bootstrap_percentile_lower, [0], 15, lw=2.5, linestyle='dashed', color='C1')

ax.hist(dart_bootstrap_train_accuracies, bins=7,
        color='#0080ff', edgecolor="none", 
        alpha=0.3)
plt.legend(loc='upper left')
plt.xlim([0.69, 0.75])
plt.tight_layout()
plt.savefig('figures/dart-bootstrap-ci-histo.svg')
# plt.show()
plt.clf()

fig, ax = plt.subplots(figsize=(8, 4))
ax.vlines( goss_bootstrap_train_mean, [0], 80, lw=2.5, linestyle='-', label='goss bootstrap train mean')
ax.vlines(goss_bootstrap_percentile_upper, [0], 15, lw=2.5, linestyle='dashed', 
          label='CI95 bootstrap, percentile', color='C1')
ax.vlines(goss_bootstrap_percentile_lower, [0], 15, lw=2.5, linestyle='dashed', color='C1')

ax.hist(goss_bootstrap_train_accuracies, bins=7,
        color='#0080ff', edgecolor="none", 
        alpha=0.3)
plt.legend(loc='upper left')
plt.xlim([0.69, 0.75])
plt.tight_layout()
plt.savefig('figures/goss-bootstrap-ci-histo.svg')
# plt.show()
plt.clf()






# lgbm_cv_acc = bootstrap_point632_score(estimator=lgbm,
#                                   X=X_train,
#                                   y=y_train,
#                                 #   method='.632+', # this caused a memory error for me (Roshan)
#                                   method='oob',
#                                   n_splits=20,
#                                   random_seed=99)
# print('LGBM OOB Bootstrap Accuracy: %.2f%%' % (np.mean(lgbm_cv_acc)*100))

# dart = lgb.LGBMClassifier(num_class=3, boosting_type='dart', class_weight='balanced', importance_type='gain', min_split_gain=.25, subsample=.9, subsample_freq=1, feature_fraction=.9, random_state=42, n_jobs=-1, silent=True)
# dart_cv_acc = bootstrap_point632_score(estimator=dart,
#                                   X=X_train,
#                                   y=y_train,
#                                 #   method='.632+', # this caused a memory error for me (Roshan)
#                                   method='oob',
#                                   n_splits=20,
#                                   random_seed=99)
# print('DART OOB Bootstrap Accuracy: %.2f%%' % (np.mean(dart_cv_acc)*100))

# dart = lgb.LGBMClassifier(num_class=3, boosting_type='goss', class_weight='balanced', importance_type='gain', min_split_gain=.25, subsample=.9, subsample_freq=1, feature_fraction=.9, random_state=42, n_jobs=-1, silent=True)
# dart_cv_acc = bootstrap_point632_score(estimator=dart,
#                                   X=X_train,
#                                   y=y_train,
#                                 #   method='.632+', # this caused a memory error for me (Roshan)
#                                   method='oob',
#                                   n_splits=20,
#                                   random_seed=99)
# print('GOSS OOB Bootstrap Accuracy: %.2f%%' % (np.mean(dart_cv_acc)*100))